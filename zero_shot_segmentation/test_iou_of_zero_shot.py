# -*- coding: utf-8 -*-
"""run_oct2hist.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/WinetraubLab/OCT2Hist-ModelInference/blob/main/run_oct2hist.ipynb

# Overview
Use this notebook to convert an OCT image to virtual histology.

To get started,
[open this notebook in colab](https://colab.research.google.com/github/WinetraubLab/OCT2Hist-ModelInference/blob/main/run_oct2hist.ipynb) and run.
"""
import sys
import torch
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
from OCT2Hist_UseModel.utils.masking import get_sam_input_points, show_points, show_mask, mask_gel_and_low_signal

from OCT2Hist_UseModel.utils.crop import crop
from zero_shot_segmentation.zero_shot_utils.predict_mask_on_oct import predict

# from google.colab import drive
sys.path.append('./zero_shot_segmentation')
import cv2
import matplotlib.pyplot as plt

sys.path.append('./OCT2Hist_UseModel')
import os


# Define the Roboflow project URL and API key
rf_api_key= "R04BinsZcBZ6PsfKR2fP"
rf_workspace= "yolab-kmmfx"
rf_project_name = "11-16-2023-zero-shot-oct"
rf_dataset_type = "coco-segmentation" #"png-mask-semantic"
version = 3
#roboflow semantic classes
DERMIS = 1
EPIDERMIS = 2

import numpy as np
from PIL import Image, ImageDraw

def coco_mask_to_numpy(image_shape, coco_mask):
    """
    Convert COCO format segmentation mask to a NumPy array.

    Parameters:
    - image_shape: Tuple (m, n) representing the shape of the image.
    - coco_mask: List of coordinates [x1, y1, x2, y2, ..., xn, yn] in COCO format.

    Returns:
    - numpy_mask: NumPy array of shape (m, n) with True within the mask boundaries and False elsewhere.
    """
    # Create an image and draw the polygon defined by the COCO mask
    mask_image = Image.new("1", image_shape[::-1], 0)
    draw = ImageDraw.Draw(mask_image)
    draw.polygon(coco_mask, outline=1, fill=1)
    del draw

    # Convert the mask image to a NumPy array
    numpy_mask = np.array(mask_image, dtype=bool)

    return numpy_mask

# Function to download images and masks from Roboflow
def download_images_and_masks(api_key, workspace, project_name, dataset_name, version):
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    dataset = project.rf_dataset_version(version).download(rf_dataset_type, overwrite = False)
    return dataset


# Function to calculate Intersection over Union (IoU)
def calculate_iou(mask_true, mask_pred, class_id):
    intersection = np.logical_and(mask_true == class_id, mask_pred == class_id)
    union = np.logical_or(mask_true == class_id, mask_pred == class_id)

    class_iou = np.sum(intersection) / np.sum(union)
    return class_iou


# Download images and masks
dataset = download_images_and_masks(rf_api_key, rf_workspace, rf_project_name, rf_dataset_type, version)
# prepare model
DEVICE = torch.device('mps')  # 'cpu'
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "weights/sam_vit_h_4b8939.pth"  # os.path.join("weights", "sam_vit_h_4b8939.pth")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
predictor = SamPredictor(sam)


total_iou = 0
total_samples = 0
annot_dataset_dir = "zero_shot_segmentation/11/16/2023-Zero-shot-OCT-3/test/"
raw_oct_dataset_dir = "/Users/dannybarash/Library/CloudStorage/GoogleDrive-dannybarash7@gmail.com/Shared drives/Yolab - Current Projects/Yonatan/Hist Images/"
# Get the list of image files
image_files = [f for f in os.listdir(annot_dataset_dir) if f.endswith(".jpg")]

total_iou = {DERMIS:0, EPIDERMIS:0}  # IOU for each class
total_samples = 0
"/Users/dannybarash/Library/CloudStorage/GoogleDrive-dannybarash7@gmail.com/Shared drives/Yolab - Current Projects/Yonatan/Hist Images/LE-03-Slide03_Section02_yp0_A.jpg"
path_to_annotations = os.path.join(annot_dataset_dir, "_annotations.coco.json")
from pylabel import importer
dataset = importer.ImportCoco(path_to_annotations, path_to_images=annot_dataset_dir, name="zero_shot_oct")
visualize = True
for image_file in tqdm(image_files):
    gt_image_path = os.path.join(raw_oct_dataset_dir, image_file)
    image_path = os.path.join(annot_dataset_dir, image_file)
    oct_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    coco_mask = dataset.df.ann_segmentation[dataset.df.img_filename == image_file].values[0][0]
    mask_true = coco_mask_to_numpy(oct_img.shape, coco_mask)
    if visualize:
        plt.figure(figsize=(5, 5))
        plt.imshow(oct_img, cmap = "gray")
        show_mask(mask_true, plt.gca())
        plt.axis('off')
        plt.show()

    # Predict the mask using your model
    masks, masked_gel_image, crop_args = predict(image_path, predictor)

    if masks is None:
        print(f"Could not segment {image_path}.")
        continue
    mask_pred = np.zeros_like(masks[0], dtype=np.float32)
    #switch to semantic_segmentation format
    masked_gel_image = masked_gel_image[:,:,0]
    mask_pred[masked_gel_image > 0] = DERMIS
    mask_pred[masks[0]] = EPIDERMIS
    #mask_pred = cv2.resize(mask_pred, (mask_true.shape[1], mask_true.shape[0]), interpolation =  cv2.INTER_NEAREST)
    mask_true = crop(mask_true, **crop_args)

    # Calculate IoU for each class
    for class_id in [DERMIS, EPIDERMIS]:
        class_iou = calculate_iou(mask_true, mask_pred, class_id)
        total_iou[class_id] += class_iou

    total_samples += 1
    cropped_oct_image = crop(oct_img, **crop_args)
    if visualize:
        mask=masks[0]
        plt.imshow(cropped_oct_image, cmap = "gray")
        show_mask(mask_true, plt.gca())
        plt.axis('off')
        plt.show()

sum_iou_all_classes = 0
for k in [DERMIS, EPIDERMIS]:
    avg_iou_class = total_iou[k]/total_samples
    sum_iou_all_classes += total_iou[k]
    print(f"iou for {k}:{avg_iou_class}")
    # Calculate the average IoU
average_iou = sum_iou_all_classes / (total_samples * len(total_iou)) #sum all ious divided by (number of images * number of classes).
print(f"Average IoU: {average_iou}")
