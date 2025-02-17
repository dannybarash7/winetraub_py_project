import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import monai
import sys
from tqdm import tqdm
import numpy as np
import albumentations as A
# from segment_anything import sam_model_registry
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

    # def transform_img(self, img_np, box_np):
    #     if len(img_np.shape) == 2:
    #         img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    #     else:
    #         img_3c = img_np
    #     H, W, C = img_3c.shape
    #     # %% image preprocessing
    #     img_1024 = transform.resize(
    #         img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    #     ).astype(np.uint8)
    #     img_1024 = (img_1024 - img_1024.min()) / np.clip(
    #         img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    #     )  # normalize to [0, 1], (H, W, 3)
    #     # convert the shape to (3, H, W)
    #     img_1024_tensor = (
    #         torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
    #     )
    #     image_embedding = self.image_encoder(img_1024_tensor)
    #     box_np = np.array([box_np])
    #     box_1024 = box_np / np.array([W, H, W, H]) * 1024
    #     return image_embedding, box_1024


    # def forward(self, image, box):
    #     #this forward version is closer to the inference code, and is limited to batch size = 1
    #     B, H, W, C = image.shape
    #     #remove batch dim
    #     image = image.squeeze(0)
    #     image_embedding, box_1024 = self.transform_img(image, box)
    #     # do not compute gradients for prompt encoder
    #     with torch.no_grad():
    #         box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
    #         if len(box_torch.shape) == 2:
    #             box_torch = box_torch[:, None, :]  # (B, 1, 4)
    #
    #         sparse_embeddings, dense_embeddings = self.prompt_encoder(
    #             points=None,
    #             boxes=box_torch,
    #             masks=None,
    #         )
    #     low_res_masks, _ = self.mask_decoder(
    #         image_embeddings=image_embedding,  # (B, 256, 64, 64)
    #         image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
    #         sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
    #         dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
    #         multimask_output=False,
    #     )
    #     #TODO: training and inference code diverge here, try both.
    #     #from training code
    #
    #     # ori_res_masks = F.interpolate(
    #     #     low_res_masks,
    #     #     size=(image.shape[2], image.shape[3]),
    #     #     mode="bilinear",
    #     #     align_corners=False,
    #     # )
    #     # return ori_res_masks
    #
    #     #from inference code
    #     low_res_pred = torch.sigmoid(low_res_masks)  # (1, 1, 256, 256)
    #
    #     low_res_pred = F.interpolate(
    #         low_res_pred,
    #         size=(H, W),
    #         mode="bilinear",
    #         align_corners=False,
    #     )  # (1, 1, gt.shape)
    #     low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    #     medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    #
    #     return medsam_seg

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
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks

#############################################
# Dataset: Paired Domain A (grayscale) & Domain B' (RGB)
#############################################
class CrossDomainDataset(Dataset):
    def __init__(self, domain_A_paths, domain_B_paths, boxes_paths, transforms_A=None, transforms_B=None):
        """
        domain_A_paths: List of file paths for Domain A images (grayscale).
        domain_B_paths: List of file paths for Domain B' images (RGB).
        transforms_A, transforms_B: Albumentations transforms for each domain.
        """
        self.domain_A_paths = domain_A_paths
        self.domain_B_paths = domain_B_paths
        self.boxes_paths = boxes_paths
        self.transforms_A = transforms_A
        self.transforms_B = transforms_B

    def __len__(self):
        return len(self.domain_A_paths)

    def __getitem__(self, idx):
        img_a_path = self.domain_A_paths[idx]
        img_b_path = self.domain_B_paths[idx]
        box_path = self.boxes_paths[idx]

        img_A = np.load(img_a_path)
        H,W,C = img_A.shape
        img_A = self.py_to_torch(img_A)
        #TODO do i need this?
        # img_A = cv2.cvtColor(img_A, cv2.COLOR_GRAY2RGB)  # Replicate channel to make 3-channel

        img_B = np.load(img_b_path)
        img_B = self.py_to_torch(img_B)
        box = np.load(box_path)
        box_np = np.array([box])
        box_1024 = box_np / np.array([W, H, W, H]) * 1024

        return img_A, img_B, box_1024

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


@torch.no_grad()
def medsam_inference(self, img, box):

    H, W, _ = img.shape
    img_embed, box_1024 = self.transform_img(img, box, save_output=False, overwrite_output=True)
    medsam_model = self.predictor.model
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )

    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

    return medsam_seg


#############################################
# Training Loop: Teacher-Student with MedSAM
#############################################
def train_teacher_student(domain_A_paths, domain_B_paths, boxes_paths, checkpoint_path, epochs, batch_size, use_scaler, shuffle_train, num_workers,pin_memory,):
    # Create dataset and dataloader
    dataset = CrossDomainDataset(domain_A_paths, domain_B_paths, boxes_paths,
                                 transforms_A=None, transforms_B=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=pin_memory)

    # Initialize teacher and student models
    teacher = MedSAMTeacherStudent(checkpoint_path, freeze_prompt_mask=True).to(device)
    student = MedSAMTeacherStudent(checkpoint_path, freeze_prompt_mask=True).to(device)

    # Set teacher to evaluation mode
    teacher.eval()

    # Optimizer: update only the image encoder parameters of the student
    # TODO: two experiments: 1. with mask decoder 2. without
    img_mask_encdec_params = list(get_params(student.model)) #+ list( student.model.mask_decoder.parameters() )
    optimizer = optim.AdamW(img_mask_encdec_params, lr=1e-4, weight_decay=1e-4)
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")

    if use_scaler:
        scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
    else:
        print("Warning: disable gradient checkpointing for faster training.")

    for epoch in range(epochs):
        student.train()
        total_loss = 0

        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", file=sys.stdout)
        for img_A, img_B, boxes in loop:
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()

            img_A = img_A.to(device)
            img_B = img_B.to(device)

            # Teacher generates pseudo-labels from Domain B' (RGB)
            with torch.no_grad():
                # pseudo_labels = teacher(img_B,boxes_np)  # Get probabilities as pseudo-labels
                pseudo_labels = torch.sigmoid(teacher(img_B,boxes_np))  # Get probabilities as pseudo-labels
            if use_scaler:
                with (torch.cuda.amp.autocast()):
                    # Student processes Domain A (converted to 3-channel RGB)
                    student_preds = student(img_A)
                    #TODO try with both losses or with one
                    loss = ce_loss(student_preds, pseudo_labels)+seg_loss(student_preds, pseudo_labels.float())

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                student_preds = student(img_A, boxes_np)
                loss = ce_loss(student_preds, pseudo_labels)+seg_loss(student_preds, pseudo_labels.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


            optimizer.zero_grad()
            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / (loop.n + 1))

            # Update teacher's image encoder using EMA from student's image encoder
            update_teacher(teacher, student, alpha=0.99)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

    # Save the fine-tuned student model
    torch.save(student.state_dict(), "student_medsam_finetuned.pth")
    print("Training complete!")


#############################################
# Main entry point
#############################################
if __name__ == "__main__":
    # Replace these with your actual file paths for Domain A (grayscale) and Domain B' (RGB) images.
    oct_img_paths = ["/Users/dannybarash/Code/oct/medsam/zero_shot_segmentation_test_sam/training_ts/saved_oct_input_image_LE-03-Slide03_Section01_yp0_A_jpg.rf.2915d6a954036475dcd6e234053fb733.jpg.npy"]
    vhist_paths = ["/Users/dannybarash/Code/oct/medsam/zero_shot_segmentation_test_sam/training_ts/saved_vhist_input_image_LE-03-Slide03_Section01_yp0_A_jpg.rf.2915d6a954036475dcd6e234053fb733.jpg.npy"]
    boxes_paths = ["/Users/dannybarash/Code/oct/medsam/zero_shot_segmentation_test_sam/training_ts/saved_oct_box_LE-03-Slide03_Section01_yp0_A_jpg.rf.2915d6a954036475dcd6e234053fb733.jpg.npy"]
    # Path to the MedSAM checkpoint
    checkpoint_path = weights_path
    batch_size = 1
    epochs = 2
    use_scaler = False
    shuffle_train= False
    num_workers= 0
    pin_memory = False
    # Run training
    train_teacher_student(oct_img_paths, vhist_paths, boxes_paths, checkpoint_path, epochs, batch_size, use_scaler, shuffle_train, num_workers,pin_memory)
