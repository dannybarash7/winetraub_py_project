import os
import pprint
import random
import subprocess
import sys

import cv2
import monai
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from skimage import transform
from torch.utils.data import Dataset, DataLoader

# Define the path to MedSAM
medsam_path = "/cs_storage/barashd/Code/medsam/zero_shot_segmentation_test_sam/"
if medsam_path not in sys.path and os.path.exists(medsam_path):
    sys.path.insert(0, medsam_path)
from MedSAM.segment_anything import sam_model_registry

running_on_cluster = 'barashd' in os.getcwd()
weights_path = "/cs_storage/barashd/Code/medsam/zero_shot_segmentation_test_sam/MedSAM/work_dir/MedSAM/medsam_vit_b.pth"
if not running_on_cluster:
    weights_path = '/Users/dannybarash/Code/oct/medsam/MedSAM/work_dir/MedSAM/medsam_vit_b.pth'

# Device configuration and seeding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)


def start_tensorboard(log_dir):
    """Starts TensorBoard in the background."""

    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    subprocess.Popen([
        "/cs_storage/barashd/.conda/envs/pytorch-CycleGAN-and-pix2pix/bin/tensorboard",
        f"--logdir={os.path.dirname(log_dir)}", "--port=6006", "--bind_all"
    ], env=env)
    print(f"TensorBoard started at http://localhost:6006 and path {log_dir}")


class MedSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    # def forward(self, image, box):
    #     image_embedding = self.image_encoder(image)
    #     with torch.no_grad():
    #         box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
    #         if len(box_torch.shape) == 2:
    #             box_torch = box_torch[:, None, :]
    #         sparse_embeddings, dense_embeddings = self.prompt_encoder(
    #             points=None, boxes=box_torch, masks=None)
    #     low_res_masks, _ = self.mask_decoder(
    #         image_embeddings=image_embedding,
    #         image_pe=self.prompt_encoder.get_dense_pe(),
    #         sparse_prompt_embeddings=sparse_embeddings,
    #         dense_prompt_embeddings=dense_embeddings,
    #         multimask_output=False)
    #     ori_res_masks = F.interpolate(low_res_masks, size=(image.shape[2], image.shape[3]),
    #                                   mode="bilinear", align_corners=False)
    #     return ori_res_masks, image_embedding


class MedSAMTeacherStudent(nn.Module):
    def __init__(self, checkpoint_path, freeze_mask_decoder=False, freeze_prompt_mask=True):
        super(MedSAMTeacherStudent, self).__init__()
        medsam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        self.model = MedSAM(
            image_encoder=medsam_model.image_encoder,
            mask_decoder=medsam_model.mask_decoder,
            prompt_encoder=medsam_model.prompt_encoder,
        ).to(device)
        if freeze_prompt_mask:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if freeze_mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

    def forward(self, image, box, return_img_embed=False):
        image_embedding = self.model.image_encoder(image)
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None, boxes=box_torch, masks=None)
        low_res_masks, _ = self.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False)
        return low_res_masks, image_embedding


class CrossDomainDataset(Dataset):
    def __init__(self, domain_A_paths, domain_B_paths, oct_boxes_paths, vhist_boxes_paths,
                 transforms_A=None, transforms_B=None, config=None, ):
        self.domain_A_paths = domain_A_paths
        self.domain_B_paths = domain_B_paths
        self.oct_boxes_paths = oct_boxes_paths
        self.vhist_boxes_paths = vhist_boxes_paths
        self.transforms_A = transforms_A
        self.transforms_B = transforms_B
        self.config = config

    def __len__(self):
        return len(self.domain_A_paths)

    def __getitem__(self, idx):
        img_a_path = self.domain_A_paths[idx]
        img_b_path = self.domain_B_paths[idx]
        oct_box_path = None if self.oct_boxes_paths is None else self.oct_boxes_paths[idx]
        vhist_box_path = None if self.vhist_boxes_paths is None else self.vhist_boxes_paths[idx]
        fname = filename_from_index(img_a_path)

        if img_a_path.endswith(".png"):
            img_A = Image.open(img_a_path).convert("RGB")
            img_A = np.array(img_A)
            H, W, C = img_A.shape
            img_A = self.py_to_torch(img_A)
        else:
            img_A = np.load(img_a_path, allow_pickle=True)
            H, W, C = img_A.shape
            img_A = self.py_to_torch(img_A)

        if img_b_path.endswith(".png"):
            img_B = Image.open(img_b_path).convert("RGB")
            img_B = np.array(img_B)
            H, W, C = img_B.shape
            img_B = self.py_to_torch(img_B)
        else:
            img_B = np.load(img_b_path, allow_pickle=True)
            H, W, C = img_B.shape
            img_B = self.py_to_torch(img_B)

        if oct_box_path is None and vhist_box_path is None:
            # Assuming semi-supervised configuration is provided in a global config
            if self.config["semi_supervised_config"]["rect"] == "random":
                rect = generate_rectangle(image_size=(1024, 1024), batch_size=1,
                                          mode=self.config["semi_supervised_config"]["rect"]).to(device)
                rect = rect.unsqueeze(0)
                vhist_box_1024 = rect
                oct_box_1024 = rect
            else:
                raise Exception("No box files and rect mode is not random.")
        else:
            oct_box_1024 = self.load_box(H, W, oct_box_path)
            vhist_box_1024 = self.load_box(H, W, vhist_box_path)

        return img_A, img_B, oct_box_1024, vhist_box_1024, fname

    def load_box(self, H, W, box_path):
        box = np.load(box_path)
        box_np = np.array([box])
        oct_box_1024 = box_np / np.array([W, H, W, H]) * 1024
        return oct_box_1024

    # def py_to_torch(self, img_np):
    #     if len(img_np.shape) == 2:
    #         img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    #     else:
    #         img_3c = img_np
    #     img_1024 = transform.resize(img_3c, (1024, 1024), order=3,
    #                                 preserve_range=True, anti_aliasing=True).astype(np.uint8)
    #     img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), 1e-8, None)
    #     assert (np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0), "Image should be normalized to [0, 1]"
    #     img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1)
    #     return img_1024_tensor

    def py_to_torch(self, img_np):
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np
        # Use cv2.resize with cubic interpolation (cv2.INTER_CUBIC)
        # Note: cv2.resize expects the size as (width, height)
        img_resized = cv2.resize(img_3c, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        img_resized = img_resized.astype(np.uint8)

        # Normalize the image to [0, 1] using dynamic range normalization.
        img_norm = (img_resized - img_resized.min()) / np.clip(img_resized.max() - img_resized.min(), 1e-8, None)
        assert (np.max(img_norm) <= 1.0 and np.min(img_norm) >= 0.0), "Image should be normalized to [0, 1]"

        img_tensor = torch.tensor(img_norm).float().permute(2, 0, 1)
        return img_tensor


def get_params(model, freeze_mask_decoder):
    from itertools import chain
    if not freeze_mask_decoder:
        return chain(model.image_encoder.parameters(), model.mask_decoder.parameters())
    else:
        return model.image_encoder.parameters()


def update_teacher(teacher, student, freeze_mask_decoder, alpha=0.99, config=None):
    for t_param, s_param in zip(get_params(teacher.model, freeze_mask_decoder),
                                get_params(student.model, freeze_mask_decoder)):
        t_param.data = alpha * t_param.data + (1 - alpha) * s_param.data


def visualize_predictions(image, mask, box, title, epoch, fname, config):
    B, C, H, W = image.shape
    images_255 = image * 255
    for i in range(images_255.shape[0]):
        image_255 = images_255[i]
        image_np = image_255.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        mask_resized = F.interpolate(mask, [H, W], mode="bilinear", align_corners=False)
        mask_np = (mask_resized * 255).detach().cpu().numpy().astype(np.uint8)
        overlay = np.zeros_like(image_np, dtype=np.uint8)
        overlay[..., 1] = mask_np[i, ...]
        overlay[..., 0] = mask_np[i, ...]
        combined = cv2.addWeighted(image_np, 0.5, overlay, 0.5, 0)
        x1, y1, x2, y2 = map(int, box[0, :])
        cv2.rectangle(combined, (x1, y1), (x2, y2), (0, 255, 0), 2)
        filename = os.path.join(config["output_dir_path"],
                                f"{title.replace(' ', '_').lower()}_epoch_{epoch + 1}_sample_{fname[i]}.png")
        combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, combined)
        print(f"Saved {filename}")


def validate(student, teacher, val_dataloader, epoch, ce_loss, seg_loss, config, writer):
    student.eval()
    teacher.eval()
    total_val_loss = 0.0
    threshold_seg_loss = 0.0
    first_image_logged = False
    with torch.no_grad():
        for i, (img_A, img_B, oct_box, vhist_box, fname) in enumerate(val_dataloader):
            img_A, img_B = img_A.to(device), img_B.to(device)
            vhist_box_np = vhist_box.detach().cpu().numpy()
            teacher_pred_mask, teacher_image_embed = teacher(img_B, vhist_box_np)
            pseudo_labels = torch.sigmoid(teacher_pred_mask)

            oct_box_np = oct_box.detach().cpu().numpy()
            student_pred_mask, student_image_embed = student(img_A, oct_box_np)
            student_mask = torch.sigmoid(student_pred_mask)

            loss = get_loss(ce_loss, seg_loss, pseudo_labels, student_mask, config["semi_supervised_config"],
                            teacher_image_embed, student_image_embed)
            total_val_loss += loss.item()

            print(f"Validating fname {fname}")
            if epoch == 0:
                visualize_predictions(img_B, pseudo_labels, vhist_box_np[0],
                                      "Teacher Predictions", epoch, fname, config)
            limit = config.get("number_of_val_images", 0)
            if i < limit:
                visualize_predictions(img_A, student_mask, oct_box_np[0],
                                      "Student Predictions", epoch, fname, config)

            if not first_image_logged:
                writer.add_image(f'Validation/Student_pred_first_image', student_mask[0].cpu(), epoch)
                first_image_logged = True

            pseudo_labels[pseudo_labels > config["prediction_threshold"]] = 1.0
            student_mask[student_mask > config["prediction_threshold"]] = 1.0
            threshold_seg_loss += seg_loss(student_mask, pseudo_labels.float()).item()

            if not running_on_cluster:
                print(f"WARNING: breaking validation loop after {fname}")
                break

    writer.add_scalar("Validation/Total_Loss", total_val_loss / len(val_dataloader), epoch)
    writer.add_scalar("Validation/Threshold_Dice_Loss", threshold_seg_loss / len(val_dataloader), epoch)


def save_checkpoint(student, teacher, optimizer, scheduler, epoch, config):
    path = os.path.join(config["output_dir_path"], "latest_checkpoint.pth")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'student_state_dict': student.state_dict(),
        'teacher_state_dict': teacher.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, path)
    path = os.path.join(config["output_dir_path"], f"checkpoint_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'student_state_dict': student.state_dict(),
        'teacher_state_dict': teacher.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, path)



def load_checkpoint(student, teacher, optimizer, scheduler, config):
    load_path = config["load_checkpoint_path"] if os.path.exists(config["load_checkpoint_path"]) \
        else os.path.join(config["output_dir_path"], "latest_checkpoint.pth")
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path, weights_only=True,map_location=torch.device('cpu'))
        student.load_state_dict(checkpoint['student_state_dict'])
        teacher.load_state_dict(checkpoint['teacher_state_dict'])
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
    end = sample_file.find("_yp")
    return sample_file[:end]


def print_filenames_of_subsets(dataset, train_dataset, val_dataset):
    global index, key
    print(f"Training files:", flush=True)
    for index in train_dataset.indices:
        key = filename_from_index(train_dataset.dataset.domain_A_paths[index])
        print(f"{index}:{key}", flush=True)
    print(f"Validation files:", flush=True)
    for index in val_dataset.indices:
        key = filename_from_index(val_dataset.dataset.domain_A_paths[index])
        print(f"{index}:{key}", flush=True)


def print_filenames(dataset, val_dataset):
    print("Training files:")
    for index, p in enumerate(dataset.domain_A_paths):
        key = filename_from_index(p)
        print(f"{index}: {key}")
    print("Validation files:")
    for index, p in enumerate(val_dataset.domain_A_paths):
        key = filename_from_index(p)
        print(f"{index}: {key}")


def generate_rectangle(image_size, batch_size, mode="random"):
    if mode == "random":
        heights = torch.randint(256, 512, (batch_size,))
        widths = torch.randint(512, 1000, (batch_size,))
        x1 = torch.randint(0, image_size[1] - widths.max().item(), (batch_size,))
        y1 = torch.randint(0, image_size[0] - 256 - heights.max().item(), (batch_size,))
        x2 = x1 + widths
        y2 = y1 + heights
    else:  # constant
        widths = torch.full((batch_size,), 512)
        heights = torch.full((batch_size,), 256)
        x1 = torch.full((batch_size,), (image_size[1] - 512) // 2)
        y1 = torch.full((batch_size,), (image_size[0] - 256) // 2)
        x2 = x1 + widths
        y2 = y1 + heights
    return torch.stack((x1, y1, x2, y2), dim=1).float()


def get_loss(ce_loss, seg_loss, pseudo_labels, student_preds, semi_supervised_config, teacher_image_embed, student_image_embed):
    loss = torch.tensor(0.0, device=device)
    if semi_supervised_config["predict_image_embed"]:
        loss += ce_loss(teacher_image_embed, student_image_embed) + seg_loss(teacher_image_embed, student_image_embed)
    if semi_supervised_config["predict_mask_decoder"]:
        loss += ce_loss(student_preds, pseudo_labels) + seg_loss(teacher_image_embed, student_image_embed)
    return loss


def training_loop(config, ce_loss, dataloader, epochs, optimizer, seg_loss, start_epoch, student, teacher,
                  val_dataloader, scheduler, writer):
    profiler_enabled = config.get("enable_profiling", False)
    if profiler_enabled:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                config["profiler_log_dir"]
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        profiler.start()
    else:
        profiler = None
    if config["use_scaler"]:
        scaler = torch.amp.GradScaler()
    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch} start")
        student.train()
        epoch_loss, epoch_ce_loss, epoch_seg_loss = 0, 0, 0

        for i, (img_A, img_B, oct_box, vhist_box, fname) in enumerate(dataloader):
            optimizer.zero_grad()
            img_A, img_B = img_A.to(device), img_B.to(device)
            with torch.no_grad():
                teacher_pred_mask, teacher_image_embed = teacher(img_B, vhist_box)
                pseudo_labels = torch.sigmoid(teacher_pred_mask)
            if config["use_scaler"]:
                with torch.amp.autocast(device_type=str(device)):
                    student_pred_mask, student_image_embed = student(img_A, oct_box)
                    student_preds = torch.sigmoid(student_pred_mask)
                    loss = get_loss(ce_loss, seg_loss, pseudo_labels, student_preds, config["semi_supervised_config"],
                                    teacher_image_embed, student_image_embed)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                student_pred_mask, student_image_embed = student(img_A, oct_box)
                student_preds = torch.sigmoid(student_pred_mask)
                loss = get_loss(ce_loss, seg_loss, pseudo_labels, student_preds, config["semi_supervised_config"],
                                teacher_image_embed, student_image_embed)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            epoch_ce_loss += ce_loss(student_preds, pseudo_labels.float()).item()
            epoch_seg_loss += seg_loss(student_preds, pseudo_labels.float()).item()

            if profiler_enabled:
                profiler.step()

            if not running_on_cluster:
                print(f"WARNING: breaking training loop after {fname}")
                break

        if config["use_scheduler"]:
            scheduler.step(epoch_loss)

        if config["semi_supervised_config"]["update_teacher_flag"]:
            update_teacher(teacher, student, config["semi_supervised_config"]["freeze_mask_decoder"], alpha=0.99,
                           config=config)

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

        # Logging extra training details
        total_weight_magnitude, total_gradient_magnitude = 0, 0
        num_params = 0
        for name, param in student.named_parameters():
            total_weight_magnitude += param.abs().mean().item()
            num_params += 1
            if param.grad is not None:
                total_gradient_magnitude += param.grad.abs().mean().item()
        avg_weight_magnitude = total_weight_magnitude / num_params if num_params > 0 else 0
        avg_gradient_magnitude = total_gradient_magnitude / num_params if num_params > 0 else 0

        writer.add_scalar("Train/Total_Loss", epoch_loss / len(dataloader), epoch)
        writer.add_scalar("Train/Dice_Loss", epoch_seg_loss / len(dataloader), epoch)
        writer.add_scalar("Train/CE_Loss", epoch_ce_loss / len(dataloader), epoch)
        writer.add_scalar("Train/Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("Train/Avg_Weight_Magnitude", avg_weight_magnitude, epoch)
        writer.add_scalar("Train/Avg_Gradient_Magnitude", avg_gradient_magnitude, epoch)

        if (epoch + 1) % config["epochs_between_val"] == 0 or epoch == 0:
            print(f"Running validation for epoch {epoch + 1}...")
            validate(student, teacher, val_dataloader, epoch, ce_loss, seg_loss, config, writer)

        if (epoch + 1) % config["epochs_between_save"] == 0:
            save_checkpoint(student, teacher, optimizer, scheduler, epoch, config)

    if profiler_enabled:
        profiler.stop()
    save_checkpoint(student, teacher, optimizer, scheduler, epochs - 1, config)
    print("Training complete!")


def train_teacher_student(config, writer, paths):
    # Build datasets
    dataset_trn = CrossDomainDataset(
        paths["oct2hist_oct_img_paths"],
        paths["oct2hist_vhist_paths"],
        None,
        None,
        config=config)
    dataset_val = CrossDomainDataset(
        paths["wsi_oct_img_paths"],
        paths["wsi_vhist_paths"],
        paths["wsi_oct_boxes_paths"],
        paths["wsi_vhist_boxes_paths"],
        config=config)

    print_filenames(dataset_trn, dataset_val)
    loader_trn = DataLoader(dataset_trn, batch_size=config["batch_size"], shuffle=config["shuffle_train"],
                            num_workers=config["num_workers"], pin_memory=config["pin_memory"])
    loader_val = DataLoader(dataset_val, batch_size=config["batch_size"], shuffle=False,
                            num_workers=config["num_workers"], pin_memory=config["pin_memory"])

    teacher = MedSAMTeacherStudent(config["checkpoint_path"],
                                   config["semi_supervised_config"]["freeze_mask_decoder"],
                                   freeze_prompt_mask=True).to(device)
    student = MedSAMTeacherStudent(config["checkpoint_path"],
                                   config["semi_supervised_config"]["freeze_mask_decoder"],
                                   freeze_prompt_mask=True).to(device)
    teacher.eval()

    optimizer = optim.AdamW(get_params(student.model, config["semi_supervised_config"]["freeze_mask_decoder"]),
                            lr=config["init_lr"],
                            weight_decay=config["semi_supervised_config"]["weight_decay"])
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5,
                                                     patience=config["patience_scheduler"],
                                                     threshold=1e-4,
                                                     threshold_mode='rel')
    start_epoch = load_checkpoint(student, teacher, optimizer, scheduler, config)
    training_loop(config, ce_loss, loader_trn, config["epochs"], optimizer, seg_loss, start_epoch,
                  student, teacher, loader_val, scheduler, writer)


def main():
    mp.set_start_method("spawn", force=True)

    def get_sorted_file_paths(root_dir, keyword, ext):
        return sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir)
                       if keyword in f and f.endswith(f".{ext}")])

    def load_file_paths(root_dir, ext, oct_files, vhist_files):
        oct_img_paths = get_sorted_file_paths(root_dir, oct_files, ext)
        oct_boxes_paths = get_sorted_file_paths(root_dir, "saved_oct_box", ext)
        vhist_paths = get_sorted_file_paths(root_dir, vhist_files, ext)
        vhist_boxes_paths = get_sorted_file_paths(root_dir, "saved_vhist_box", ext)
        return oct_img_paths, oct_boxes_paths, vhist_paths, vhist_boxes_paths

    checkpoint_path = weights_path
    batch_size = 2
    epochs = 300
    use_scaler = running_on_cluster
    shuffle_train = True
    num_workers = batch_size*2 if running_on_cluster else 0
    pin_memory = False
    epochs_between_save = 25
    epochs_between_val = 1
    init_lr = 1e-3
    prediction_threshold = 0.5
    use_scheduler = True
    patience_scheduler = 10
    use_ce_loss = True
    profiler_enabled = False

    experiment_name = "ce_seg_loss_rectRand_predDec_weightDecay5e-4"
    output_dir_root = "/cs/cs_groups/orenfr_group/barashd/ts_training/checkpoints/"
    if not running_on_cluster:
        output_dir_root = "./checkpoints/"
    output_dir_path = os.path.join(output_dir_root, experiment_name)
    os.makedirs(output_dir_path, exist_ok=True)

    semi_supervised_config = {
        "rect": "random",
        "predict_image_embed": False,
        "predict_mask_decoder": True,
        "freeze_mask_decoder": False,
        "weight_decay": 1e-4,
        "feature_predictor_mlp": False,
        "update_teacher_flag": False
    }

    # Gather experiment configuration into one dictionary.
    experiment_config = {
        "checkpoint_path": checkpoint_path,
        "batch_size": batch_size,
        "epochs": epochs,
        "use_scaler": use_scaler,
        "shuffle_train": shuffle_train,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "epochs_between_save": epochs_between_save,
        "epochs_between_val": epochs_between_val,
        "init_lr": init_lr,
        "prediction_threshold": prediction_threshold,
        "use_scheduler": use_scheduler,
        "patience_scheduler": patience_scheduler,
        "output_dir_path": output_dir_path,
        "use_ce_loss": use_ce_loss,
        "load_checkpoint_path": "/Users/dannybarash/Downloads/experiments_25_02_25/ce_seg_loss_rectRand_predDec_weightDecay5e-4/latest_checkpoint.pth",  # set if needed
        "semi_supervised_config": semi_supervised_config,
        "profiler_enabled": profiler_enabled
    }

    # Load file paths and add them to the config.
    wsi_100pairs_root = "/home/barashd/Code/medsam/zero_shot_segmentation_test_sam/training_data"
    pix2pix_training_root = "/cs/cs_groups/orenfr_group/barashd/ts_training/TrainSetPix2Pix"
    root_tb_folder = "/cs/cs_groups/orenfr_group/barashd/ts_training/tensorboards/"
    if not running_on_cluster:
        pix2pix_training_root = "/Users/dannybarash/Code/oct/medsam/data/training_ts/TrainSetPix2Pix"
        wsi_100pairs_root = "/Users/dannybarash/Code/oct/medsam/data/training_ts/training_data"
        root_tb_folder = "/Users/dannybarash/Code/oct/medsam/data/training_ts/tensorboards"
    os.makedirs(root_tb_folder, exist_ok=True)

    wsi_oct_img_paths, wsi_oct_boxes_paths, wsi_vhist_paths, wsi_vhist_boxes_paths = load_file_paths(
        wsi_100pairs_root, ext="npy",
        oct_files="saved_oct_input_image",
        vhist_files="saved_vhist_input_image")
    oct2hist_oct_img_paths, _, oct2hist_vhist_paths, _ = load_file_paths(
        pix2pix_training_root, ext="png",
        oct_files="real_A",
        vhist_files="fake_B")

    print(f"Found {len(oct2hist_oct_img_paths) + len(wsi_oct_img_paths)} OCT images")
    print(f"Found {len(wsi_oct_boxes_paths)} OCT boxes")
    print(f"Found {len(oct2hist_vhist_paths) + len(wsi_vhist_paths)} VHist images")
    print(f"Found {len(wsi_vhist_boxes_paths)} VHist boxes")

    from datetime import datetime
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H:%M")
    tb_log_dir = os.path.join(root_tb_folder, experiment_name, experiment_name + "_" + time_str)
    tb_profiler_log_dir = os.path.join(root_tb_folder, experiment_name, "profiler_log_" + "_" + time_str)
    if running_on_cluster:
        os.makedirs(tb_log_dir, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=tb_log_dir)
    if running_on_cluster:
        start_tensorboard(tb_log_dir)

    # Add file paths and the writer into the experiment configuration.
    experiment_config.update({
        "profiler_log_dir": tb_profiler_log_dir
    })

    paths_dict = {
        "wsi_oct_img_paths": wsi_oct_img_paths,
        "wsi_oct_boxes_paths": wsi_oct_boxes_paths,
        "wsi_vhist_paths": wsi_vhist_paths,
        "wsi_vhist_boxes_paths": wsi_vhist_boxes_paths,
        "oct2hist_oct_img_paths": oct2hist_oct_img_paths,
        "oct2hist_vhist_paths": oct2hist_vhist_paths, }

    # Save the experiment configuration to a file.
    config_file_path = os.path.join(output_dir_path, "experiment_config.txt")
    with open(config_file_path, "w") as file:
        for key, value in experiment_config.items():
            file.write(f"{key}: {value}\n")

    print(f"Experiment configuration saved to {config_file_path}")
    print(f"Experiment name: {experiment_name}")
    print("Experiment config:")
    print("==================")
    pprint.pprint(experiment_config)

    train_teacher_student(experiment_config, writer, paths_dict)
    writer.close()


if __name__ == "__main__":
    main()
