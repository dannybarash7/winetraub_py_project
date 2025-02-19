import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import monai
import sys
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import random
from skimage import transform
from MedSAM.segment_anything import sam_model_registry

weights_path = '/Users/dannybarash/Code/oct/medsam/MedSAM/work_dir/MedSAM/medsam_vit_b.pth'
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42  # Choose any fixed number
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using CUDA
random.seed(seed)
np.random.seed(seed)

class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks

#############################################
# Wrapper: Freeze prompt encoder and mask decoder
#############################################
class MedSAMTeacherStudent(nn.Module):
    def __init__(self, checkpoint_path, freeze_prompt_mask=True):
        """
        Initializes MedSAM from a given checkpoint.
        If freeze_prompt_mask is True, the prompt encoder and mask decoder are frozen.
        """
        super(MedSAMTeacherStudent, self).__init__()
        medsam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        self.model = MedSAM(
            image_encoder=medsam_model.image_encoder,
            mask_decoder=medsam_model.mask_decoder,
            prompt_encoder=medsam_model.prompt_encoder,
        ).to(device)
        if freeze_prompt_mask:
            # Freeze all parameters in the prompt encoder
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
            # Freeze all parameters in the mask decoder
            # for param in self.model.mask_decoder.parameters():
            #     param.requires_grad = False
        # Only the image encoder remains trainable.

    def forward(self, image, box):
        image_embedding = self.model.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.model.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        # ori_res_masks = F.interpolate(
        #     low_res_masks,
        #     size=(image.shape[2], image.shape[3]),
        #     mode="bilinear",
        #     align_corners=False,
        # )
        return low_res_masks

#############################################
# Dataset: Paired Domain A (grayscale) & Domain B' (RGB)
#############################################
class CrossDomainDataset(Dataset):
    def __init__(self, domain_A_paths, domain_B_paths, oct_boxes_paths, vhist_boxes_paths, transforms_A=None, transforms_B=None):
        """
        domain_A_paths: List of file paths for Domain A images (grayscale).
        domain_B_paths: List of file paths for Domain B' images (RGB).
        transforms_A, transforms_B: Albumentations transforms for each domain.
        """
        self.domain_A_paths = domain_A_paths
        self.domain_B_paths = domain_B_paths
        self.oct_boxes_paths = oct_boxes_paths
        self.vhist_boxes_paths = vhist_boxes_paths
        self.transforms_A = transforms_A
        self.transforms_B = transforms_B

    def __len__(self):
        return len(self.domain_A_paths)

    def __getitem__(self, idx):
        img_a_path = self.domain_A_paths[idx]
        img_b_path = self.domain_B_paths[idx]
        oct_box_path = self.oct_boxes_paths[idx]
        vhist_box_path = self.oct_boxes_paths[idx]
        fname = filename_from_index(img_a_path)
        img_A = np.load(img_a_path)
        H,W,C = img_A.shape
        img_A = self.py_to_torch(img_A)
        #TODO do i need this?
        # img_A = cv2.cvtColor(img_A, cv2.COLOR_GRAY2RGB)  # Replicate channel to make 3-channel

        img_B = np.load(img_b_path)
        img_B = self.py_to_torch(img_B)

        oct_box_1024 = self.load_box(H, W, oct_box_path)

        vhist_box_1024 = self.load_box(H, W, vhist_box_path)

        return img_A, img_B, oct_box_1024, vhist_box_1024,fname

    def load_box(self, H, W, oct_box_path):
        box = np.load(oct_box_path)
        box_np = np.array([box])
        oct_box_1024 = box_np / np.array([W, H, W, H]) * 1024
        return oct_box_1024

    def py_to_torch(self, img_np):
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np

        img_1024 = transform.resize(
            img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )
        assert (
                np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).to(device)
        )
        return img_1024_tensor


#############################################
# Data Transforms (using ImageNet stats for RGB)
#############################################

def get_params(model):
    return model.image_encoder.parameters()#+model.mask_decoder.parameters()
#############################################
# EMA Update: Update teacher's image encoder from student's
#############################################
def update_teacher(teacher, student, alpha=0.99):
    # Update only the image encoder parameters using EMA
    for t_param, s_param in zip(get_params(teacher.model), get_params(student.model)):
        t_param.data = alpha * t_param.data + (1 - alpha) * s_param.data

import matplotlib.pyplot as plt


def visualize_predictions(image, mask, box, title, epoch,fname):
    """
    Saves the predictions with a semi-transparent blue mask to a directory.
    """
    B,C, H, W = image.shape
    image_255 = image * 255
    image_np=image_255.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    mask = F.interpolate(
        mask,
        [H,W],
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    mask_np = (mask * 255).detach().cpu().numpy().astype(np.uint8)
    overlay = np.zeros_like(image_np, dtype=np.uint8)
    overlay[...,1] = mask_np # green channel
    overlay[...,0] = mask_np # blue channel
    combined = cv2.addWeighted(image_np, 0.5, overlay, 0.5, 0)

    # Draw the box
    x1, y1, x2, y2 = map(int, box[0,:])
    cv2.rectangle(combined, (x1, y1), (x2, y2), (0, 255, 0), 2)

    filename = os.path.join(f"{checkpoint_dir}",f"{title.replace(' ', '_').lower()}_epoch_{epoch+1}_sample_{fname}.png")
    combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, combined)


def validate(student, teacher, val_dataloader, epoch):
    """
    Runs validation and visualizes predictions.
    """
    student.eval()
    teacher.eval()
    with torch.no_grad():#img_A, img_B, oct_box, vhist_box
        for img_A, img_B, oct_box, vhist_box, fname in val_dataloader:
            img_A, img_B = img_A.to(device), img_B.to(device)
            vhist_box_np = vhist_box.detach().cpu().numpy()

            # Teacher predictions
            teacher_preds = torch.sigmoid(teacher(img_B, vhist_box_np))

            # Student predictions
            oct_box_np = oct_box.detach().cpu().numpy()

            student_preds = torch.sigmoid(student(img_A, oct_box_np))

            # Convert images to numpy
            # img_A_np = img_A.squeeze().permute(1, 2, 0).cpu().numpy()
            # img_B_np = img_B.squeeze().permute(1, 2, 0).cpu().numpy()
            print(f"Validating fname {fname}")
            # Visualize predictions
            visualize_predictions(img_B, teacher_preds, vhist_box_np[0], "Teacher Predictions",epoch,fname)
            visualize_predictions(img_A, student_preds, oct_box_np[0], "Student Predictions",epoch,fname)

            # Visualize thresholded predictions
            teacher_preds[teacher_preds > prediction_threshold] = 1.0
            student_preds[student_preds > prediction_threshold] = 1.0

            visualize_predictions(img_B, teacher_preds, vhist_box_np[0], "Teacher Predictions (threshold)",epoch,fname)
            visualize_predictions(img_A, student_preds, oct_box_np[0], "Student Predictions (threshold)",epoch,fname)

def save_checkpoint(student, teacher, optimizer, scheduler, epoch):
    """
    Saves the student model, teacher model, and optimizer state to resume training.
    """
    path = os.path.join(checkpoint_dir, f"latest_checkpoint.pth")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'student_state_dict': student.state_dict(),
        'teacher_state_dict': teacher.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, path)


def load_checkpoint(student, teacher, optimizer, scheduler):
    """
    Loads the student model, teacher model, and optimizer state from a checkpoint.
    """
    path = os.path.join(checkpoint_dir, f"latest_checkpoint.pth")
    if os.path.exists(path):
        checkpoint = torch.load(path)
        student.load_state_dict(checkpoint['student_state_dict'])
        teacher.load_state_dict(checkpoint['teacher_state_dict'])  # Load exact EMA teacher state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        start_epoch = 0
        print("No checkpoint found, starting from scratch.")

    return start_epoch

def filename_from_index(sample):
    sample_file = os.path.basename(sample)
    key = "_".join(sample_file.split("_")[4:])
    i = key.find("_jpg")
    return key[:i]

def print_filenames(dataset, train_dataset, val_dataset):
    global index, key
    print(f"Training files:",flush=True)
    for index in train_dataset.indices:
        key = filename_from_index(train_dataset.dataset.domain_A_paths[index])
        print(f"{index}:{key}",flush=True)
    print(f"Validation files:",flush=True)
    for index in val_dataset.indices:
        key = filename_from_index(val_dataset.dataset.domain_A_paths[index])
        print(f"{index}:{key}", flush=True)


def train_teacher_student(domain_A_paths, domain_B_paths, oct_boxes_paths, vhist_boxes_paths, checkpoint_path, epochs, batch_size, use_scaler,
                          shuffle_train, num_workers, pin_memory, epochs_between_val, epochs_between_save):
    dataset = CrossDomainDataset(domain_A_paths, domain_B_paths, oct_boxes_paths, vhist_boxes_paths)

    # Split into training and validation
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print_filenames(dataset, train_dataset, val_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers,
                            pin_memory=pin_memory)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory)
    # print("Warning: setting validation dataset to the training dataset")
    # val_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    #                         pin_memory=pin_memory)

    teacher = MedSAMTeacherStudent(checkpoint_path, freeze_prompt_mask=True).to(device)
    student = MedSAMTeacherStudent(checkpoint_path, freeze_prompt_mask=True).to(device)
    teacher.eval()
    # Optimizer: update only the image encoder parameters of the student
    # TODO: two experiments: 1. with mask decoder 2. without
    # img_mask_encdec_params = list(get_params(student.model)) #+ list( student.model.mask_decoder.parameters() )
    # optimizer = optim.AdamW(img_mask_encdec_params, lr=1e-4, weight_decay=1e-4)
    optimizer = optim.AdamW(get_params(student.model), lr=init_lr, weight_decay=1e-4)
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # Because you want to reduce when error doesn't improve
        factor=0.5,  # Reduce LR by multiplying with this factor
        patience=patience_scheduler,  # Wait for 15 epochs before reducing LR
        threshold=1e-4,  # Minimum improvement required to reset patience
        threshold_mode='rel',  # Relative to previous best
        verbose=True  # Print LR updates
    )  # Move optimizer state tensors to the correct device after loading state dict

    start_epoch = load_checkpoint(student, teacher, optimizer, scheduler)
    if not use_scaler:
        print("Warning: disable gradient checkpointing for faster training.")

    training_loop(ce_loss, train_dataloader, epochs, optimizer, seg_loss, start_epoch, student, teacher, use_scaler,
                  val_dataloader,scheduler, epochs_between_val, epochs_between_save)


def training_loop(ce_loss, dataloader, epochs, optimizer, seg_loss, start_epoch, student, teacher, use_scaler,
                  val_dataloader,scheduler, epochs_between_val, epochs_between_save):
    for epoch in range(start_epoch, epochs):
        student.train()
        epoch_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", file=sys.stdout)
        # for img_A, img_B, oct_box, vhist_box,fname in loop:
        #     optimizer.zero_grad()
        #     img_A, img_B = img_A.to(device), img_B.to(device)
        #     oct_box_np = oct_box.detach().cpu().numpy()
        #     vhist_box_np = vhist_box.detach().cpu().numpy()
        #
        #     img_A = img_A.to(device)
        #     img_B = img_B.to(device)
        #
        #     # Teacher generates pseudo-labels from Domain B' (RGB)
        #     with torch.no_grad():
        #         # pseudo_labels = teacher(img_B,boxes_np)  # Get probabilities as pseudo-labels
        #         pseudo_labels = torch.sigmoid(teacher(img_B, vhist_box_np))  # Get probabilities as pseudo-labels
        #     if use_scaler:
        #         scaler = torch.cuda.amp.GradScaler()
        #         with (torch.cuda.amp.autocast()):
        #             # Student processes Domain A (converted to 3-channel RGB)
        #             student_preds = torch.sigmoid(student(img_A, oct_box_np))
        #             # TODO try with both losses or with one
        #             loss = ce_loss(student_preds, pseudo_labels) + seg_loss(student_preds, pseudo_labels.float())
        #
        #         scaler.scale(loss).backward()
        #         scaler.step(optimizer)
        #         scaler.update()
        #     else:
        #         student_preds = torch.sigmoid(student(img_A, oct_box_np))
        #         loss = ce_loss(student_preds, pseudo_labels) + seg_loss(student_preds, pseudo_labels.float())
        #         loss.backward()
        #         optimizer.step()
        #         optimizer.zero_grad()
        #
        #     optimizer.zero_grad()
        #     epoch_loss += loss.item()
        #     loop.set_postfix(loss=epoch_loss / (loop.n + 1))
        #     if use_scheduler:
        #         scheduler.step(epoch_loss)
        #
        #     # Update teacher's image encoder using EMA from student's image encoder
        #     update_teacher(teacher, student, alpha=0.99)

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

        # Run validation every 10 epochs
        if (epoch + 1) % epochs_between_val == 0 or epoch == 0:
            print(f"Running validation for epoch {epoch + 1}...")
            validate(student, teacher, val_dataloader,epoch)
        print(f"Warning: disabled checkpoint saving")
        # if (epoch + 1) % epochs_between_save == 0:
        #     save_checkpoint(student, teacher, optimizer, scheduler, epoch)
    # save_checkpoint(student, teacher, optimizer, scheduler, epochs - 1)
    print("Training complete!")


#############################################
# Main entry point
#############################################
if __name__ == "__main__":
    def get_sorted_file_paths(root_dir, keyword):
        """
        Retrieves all files in the root directory that match a given keyword and sorts them.
        """
        return sorted(
            [os.path.join(root_dir, f) for f in os.listdir(root_dir) if keyword in f and f.endswith(".npy")]
        )


    def load_file_paths(root_dir):
        """
        Traverses the root directory and assigns files to the corresponding arrays.
        """
        oct_img_paths = get_sorted_file_paths(root_dir, "saved_oct_input_image")
        oct_boxes_paths = get_sorted_file_paths(root_dir, "saved_oct_box")

        vhist_paths = get_sorted_file_paths(root_dir, "saved_vhist_input_image")
        vhist_boxes_paths = get_sorted_file_paths(root_dir, "saved_vhist_box")

        return oct_img_paths, oct_boxes_paths, vhist_paths, vhist_boxes_paths



    # Path to the MedSAM checkpoint
    checkpoint_path = weights_path
    batch_size = 1
    epochs = 1
    use_scaler = False
    shuffle_train= False
    num_workers= 0
    pin_memory = False
    epochs_between_save = 10
    epochs_between_val = 5
    init_lr = 1e-3
    prediction_threshold = 0.5

    use_scheduler = False
    patience_scheduler = 10

    checkpoint_dir = "/cs/cs_groups/orenfr_group/barashd/ts_training/checkpoints/18.2"
    if not os.path.exists(checkpoint_dir):
        checkpoint_dir = "./checkpoints/18.2"
    # checkpoint_path =""
    root_folder = "/home/barashd/Code/pytorch-CycleGAN-and-pix2pix/ts_training/data_of_oct"
    if not os.path.exists(root_folder):
        root_folder = "/Users/dannybarash/Code/oct/medsam/data/training_ts/training_data"

    # Load the paths
    oct_img_paths, oct_boxes_paths, vhist_paths, vhist_boxes_paths = load_file_paths(root_folder)
    os.makedirs(f"{checkpoint_dir}", exist_ok=True)

    # Print results for verification
    print(f"Found {len(oct_img_paths)} OCT images")
    print(f"Found {len(oct_boxes_paths)} OCT boxes")
    print(f"Found {len(vhist_paths)} VHist images")
    print(f"Found {len(vhist_boxes_paths)} VHist boxes")

    train_teacher_student(oct_img_paths, vhist_paths, oct_boxes_paths, vhist_boxes_paths, checkpoint_path, epochs, batch_size, use_scaler, shuffle_train, num_workers, pin_memory, epochs_between_val, epochs_between_save)
