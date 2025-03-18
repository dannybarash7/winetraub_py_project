import json
import os
import shutil

import numpy as np
import cv2
import random

import torch
from libcpab.cpab import Cpab

from zero_shot_segmentation.zero_shot_utils.utils import extract_oct_base_name

tess_size = (3,3)
# No to manage device in libcpab "sample_transformation_with_prior" function.
# for now use cpu
device = "cpu"
zero_boundary = True
volume_perservation = True

config =  {
        "translation_y_min": 100,
        "translation_y_max": 200,
        "shear_range": 25,
        "rotation_range": 40,
        "contrast_alpha": 80,
        "contrast_beta": 80,
        "mirroring_probability": 0.5,
        "mean_intensity_min": 30,
        "mean_intensity_max": 50,
        "cpab": {
            "enabled": True,
            "length_scale": 0.1,
            "output_variance": 0.4,
            "tess_size": [
                3,
                3
            ]
        },
        "debug": True
    }


# config = {
#     "translation_range": 100,          # Maximum translation (pixels) in each direction
#     "shear_range": 20,                  # Maximum shear angle (degrees)
#     "rotation_range": 20,               # Maximum rotation angle (degrees)
#     "contrast_alpha": 80.0,            # Gamma distribution shape parameter for contrast
#     "contrast_beta": 80.0,             # Gamma distribution rate parameter for contrast (scale = 1 / beta)
#     "mirroring_probability": 0.4,      # Probability to flip image left-right
#     "cpab": {
#         "enabled": True,               # Whether to perform the CPAB augmentation
#         "length_scale": 0.1,           # Length scale parameter for the CPAB transform
#         "output_variance": 0.4,        # Output variance for the CPAB transform
#         "tess_size": (3, 3)            # Tessellation size for the CPAB transform
#     }
# }


def transform_with_theta(image,theta):
    img_tensor = torch.tensor(image)
    img_input = img_tensor.unsqueeze(0).permute(0, 3, 2,1).to(device)

    T = Cpab(tess_size, device, zero_boundary, volume_perservation)
    B, C, W, H = img_input.shape
    outsize = (W, H)
    img_warp = T.transform_data(img_input, theta, outsize=outsize)
    img_warp = img_warp.squeeze().permute(2, 1, 0).detach().cpu().numpy()
    if image.max() > 1.0:
        img_warp = img_warp.astype(np.uint8)
    return img_warp

def sample_with_prior(img, mask, dont_care_mask, length_scale=0.1, output_variance=0.2 ,tess_size=(3,3)):
    # img_tensor = torch.tensor(img)
    # mask_tensor = torch.tensor(mask)
    # dont_care_mask_tensor = torch.tensor(dont_care_mask)

    # sample warp
    T = Cpab(tess_size, device, zero_boundary, volume_perservation)

    theta = T.sample_transformation_with_prior(n_sample=1, mean=None,
                                          length_scale=length_scale, output_variance=output_variance)
    
    # prepare image
    # img_input = img_tensor.unsqueeze(0).permute(0, 3, 2,1).to(device)
    # mask_input = mask_tensor.unsqueeze(0).permute(0, 3, 2,1).to(device)
    # dont_care_mask_input = dont_care_mask_tensor.unsqueeze(0).permute(0, 3, 2,1).to(device)
    # add batch dim
    # B, C, W, H = img_input.shape
    # outsize = (W, H)
    # img_warp = T.transform_data(img_input, theta, outsize=outsize)
    # mask_warp = T.transform_data(mask_input, theta, outsize=outsize)
    # dont_care_mask_warp = T.transform_data(dont_care_mask_input, theta, outsize=outsize)
    # img_warp = img_warp.squeeze().permute(2,1,0).detach().cpu().numpy()
    # mask_warp = mask_warp.squeeze().permute(2,1,0).detach().cpu().numpy()
    # dont_care_mask_warp = dont_care_mask_warp.squeeze().permute(2,1,0).detach().cpu().numpy()
    # remove batch dim, channel first for plotting
    # img_warp = img_warp.astype(np.uint8)
    
    return theta


def load_data(root_dir):
    data = {}


    # Load images
    for filename in os.listdir(root_dir):
        if not filename.startswith("gt_mask") and not filename.startswith("dont_care") and "oct_" in filename:  # Ensure only oct images are processed
            # name = filename.replace("_cropped_oct_image.png", "")
            base_name = extract_oct_base_name(filename, "oct_")
            print(f"Loaded based name oct image {base_name}")

            image_path = os.path.join(root_dir, filename)

            if base_name not in data:
                data[base_name] = {}

            data[base_name]["oct"] = image_path

        if filename.startswith("gt_mask_") and filename.endswith(".npy"):  # Ensure only masks are processed
            base_name = extract_oct_base_name(filename, "gt_mask_")
            print(f"Loaded based name gt mask {base_name}")

            mask_path = os.path.join(root_dir, filename)


            if base_name not in data:
                data[base_name] = {}

            data[base_name]["mask"] = mask_path

        if filename.startswith("dont_care_") and filename.endswith(".npy"):  # Ensure only masks are processed
            base_name = extract_oct_base_name(filename, "dont_care_")
            mask_path = os.path.join(root_dir, filename)


            if base_name not in data:
                data[base_name] = {}

            data[base_name]["dont_care_mask"] = mask_path


    if config["debug"]:
        for base_name, items in data.items():
            assert len(items)==3
    return data


def random_augment(image, mask, dont_care_mask, image_name, augment_mask=False, output_dir=None):
    h, w, _ = image.shape
    log = {"image_name": image_name, "augmentations": []}


    # Random translation
    tx = 0
    ty = random.randint(config["translation_y_min"], config["translation_y_max"])
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, translation_matrix, (w, h))
    if augment_mask:
        mask = cv2.warpAffine(mask, translation_matrix, (w, h))
        dont_care_mask = cv2.warpAffine(dont_care_mask, translation_matrix, (w, h))
    log["augmentations"].append({"type": "translation", "tx": tx, "ty": ty})

    # Global shearing
    if config["shear_range"] > 0:
        shear = random.uniform(-config["shear_range"], config["shear_range"])
        shear_matrix = np.float32([[1, np.tan(np.radians(shear)), 0], [0, 1, 0]])
        image = cv2.warpAffine(image, shear_matrix, (w, h))
        if augment_mask:
            mask = cv2.warpAffine(mask, shear_matrix, (w, h))
            dont_care_mask = cv2.warpAffine(dont_care_mask, shear_matrix, (w, h))
        log["augmentations"].append({"type": "shear", "angle": shear})

    # Small rotations
    if config["rotation_range"]>0:
        angle = random.uniform(-config["rotation_range"], config["rotation_range"])
        rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        image = cv2.warpAffine(image, rotation_matrix, (w, h))
        if augment_mask:
            mask = cv2.warpAffine(mask, rotation_matrix, (w, h))
            dont_care_mask = cv2.warpAffine(dont_care_mask, rotation_matrix, (w, h))
        log["augmentations"].append({"type": "rotation", "angle": angle})

    # Contrast changes
    contrast_add = 1 + random.uniform(config["mean_intensity_min"],config["mean_intensity_max"])
    image = np.clip(image + contrast_add, 0, 255).astype(np.uint8)
    log["augmentations"].append({
        "type": "contrast_add",
        "factor": contrast_add,
    })

    alpha = config["contrast_alpha"]
    beta = config["contrast_beta"]
    contrast_factor = np.random.gamma(alpha, 1 / beta)
    image = np.clip(image * contrast_factor, 0, 255).astype(np.uint8)
    # image2 = np.clip(image * contrast_factor, 0, 255).astype(np.uint8)
    log["augmentations"].append({
        "type": "contrast_factor",
        "factor": contrast_factor,
        "alpha": alpha,
        "beta": beta
    })

    # Mirroring (left-right)
    if random.random() < config["mirroring_probability"]:
        image = cv2.flip(image, 1)
        if augment_mask:
            mask = cv2.flip(mask, 1)
            dont_care_mask = cv2.flip(dont_care_mask, 1)
        log["augmentations"].append({"type": "mirroring", "axis": "left-right"})

    # CPAB transformation (if enabled)
    if config["cpab"]["enabled"]:
        # These functions (sample_with_prior and transform_with_theta) are assumed to be defined elsewhere.
        theta = sample_with_prior(
            image, mask, dont_care_mask,
            length_scale=config["cpab"]["length_scale"],
            output_variance=config["cpab"]["output_variance"],
            tess_size=config["cpab"]["tess_size"]
        )
        image = transform_with_theta(image, theta)
        if augment_mask:
            mask = transform_with_theta(mask, theta)
            dont_care_mask = transform_with_theta(dont_care_mask, theta)
        log["augmentations"].append({
            "type": "sample_with_prior",
            "length_scale": config["cpab"]["length_scale"],
            "output_variance": config["cpab"]["output_variance"],
            "tess_size": config["cpab"]["tess_size"],
            "theta": theta.tolist()
        })

    # Save the log to the output directory, if provided
    log_general.append(log)
    print(log)

    if augment_mask:
        return image, (mask > 0.5).astype(np.float32), (dont_care_mask > 0.5).astype(np.float32)
    else:
        return image, None, None

def augment_and_save(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    #
    mask, dont_care_mask=None,None
    one_channel_image=False
    one_channel_mask=False
    one_channel_dontcare=False
    for base_name, items in data.items():
        image = cv2.imread(items["oct"], cv2.IMREAD_GRAYSCALE)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
            one_channel_image = True
        if "mask" in items:
            mask = np.load(items["mask"], allow_pickle=True).astype(np.float32)  # Convert mask to float
            if len(mask.shape) == 2:
                mask = np.stack([mask] * 3, axis=-1)
                one_channel_mask = True
            dont_care_mask = np.zeros_like(mask)
        if "dont_care_mask" in items:
            dont_care_mask = np.load(items["dont_care_mask"], allow_pickle=True).astype(np.float32)  # Convert mask to float
            if len(dont_care_mask.shape) == 2:
                dont_care_mask = np.stack([dont_care_mask] * 3, axis=-1)
                one_channel_dontcare = True

        aug_img, aug_mask, aug_dont_care_mask = random_augment(image, mask,dont_care_mask, base_name, augment_mask=True, output_dir=output_dir)
        if one_channel_image:
            aug_img = aug_img[:,:,0]
        if one_channel_mask:
            aug_mask = aug_mask[:,:,0]
        aug_mask = aug_mask>0.5
        if one_channel_dontcare:
            aug_dont_care_mask = aug_dont_care_mask[:,:,0]
        aug_dont_care_mask = aug_dont_care_mask>0.5
        cv2.imwrite(os.path.join(output_dir, f"oct_{base_name}.png"), aug_img)
        np.save(os.path.join(output_dir, f"gt_mask_{base_name}.npy"), aug_mask)
        np.save(os.path.join(output_dir, f"dont_care_{base_name}.npy"), aug_dont_care_mask)

if __name__ == "__main__":
    # data = load_data("/Users/dannybarash/Documents/university/oct/augmentations/augmented_dataset/before")
    # data = load_data("/Users/dannybarash/Code/oct/medsam/zero_shot_segmentation_test_sam/images/dataset_v7/candidate_a")
    data = load_data("/Users/dannybarash/Code/oct/medsam/zero_shot_segmentation_test_sam/images/dataset_v7/dontcare_fixed/debugging_augmentation_v2/augmentation_data")
    # output_dir = "/Users/dannybarash/Documents/university/oct/augmentations/augmented_dataset/augments_05"
    # output_dir = "/Users/dannybarash/Documents/university/oct/augmentations/augmented_dataset/augments_06"
    output_dir = "/Users/dannybarash/Documents/university/oct/augmentations/augmented_dataset/augments_07"
    # output_dir = "/Users/dannybarash/Documents/university/oct/augmentations/augmented_dataset/delicate"
    # output_dir = "/Users/dannybarash/Documents/university/oct/augmentations/augmented_dataset/stronger"
    # output_dir = "/Users/dannybarash/Documents/university/oct/augmentations/augmented_dataset/stronger_with_rotations"
    # output_dir = "/Users/dannybarash/Documents/university/oct/augmentations/augmented_dataset/debugging_augmentation_v2"
    # output_dir = "/Users/dannybarash/Documents/university/oct/augmentations/augmented_dataset/after_cpab_04_and_increased_params"
    # output_dir = "/Users/dannybarash/Documents/university/oct/augmentations/augmented_dataset/after_cpab_04"
    # output_dir = "/Users/dannybarash/Documents/university/oct/augmentations/augmented_dataset/after"
    log_path = os.path.join(output_dir, "log.json")
    log_general = []
    if os.path.exists(log_path):
        os.remove(log_path)
    log_general.append(config)
    print(f"Output directory: {output_dir}")
    print(config)
    augment_and_save(data, output_dir)
    if output_dir is not None:
        print(log_general)  # Logging the augmentations to the console
        with open(log_path, "w") as log_file:
            json.dump(log_general, log_file, indent=4)
        print(f"Saved log to {log_path}")