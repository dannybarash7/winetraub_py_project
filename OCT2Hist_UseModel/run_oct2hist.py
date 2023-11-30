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

import random
import sys
import time

import numpy as np

from utils.show_images import showImg

import torch
from segment_anything import sam_model_registry, SamPredictor

import oct2hist
from OCT2Hist_UseModel.utils.masking import get_sam_input_points, show_points, show_mask, mask_gel_and_low_signal

# from google.colab import drive

sys.path.append('./zero_shot_segmentation')
import cv2
import matplotlib.pyplot as plt

sys.path.append('./OCT2Hist_UseModel')
import os

DEVICE = torch.device('mps')  # 'cpu'
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "weights/sam_vit_h_4b8939.pth"  # os.path.join("weights", "sam_vit_h_4b8939.pth")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
predictor = SamPredictor(sam)

# pick image
image_directory = '/Users/dannybarash/Downloads/TestSet/'
all_images = os.listdir(image_directory)
filtered_images = [img for img in all_images if "real_A" in img]
random.shuffle(filtered_images)
filtered_images = filtered_images[:100]


def measure_performance():
    pass


for filename in filtered_images:
    # oct_input_image_path = "/Users/dannybarash/Library/CloudStorage/GoogleDrive-dannybarash7@gmail.com/Shared drives/Yolab - Current Projects/_Datasets/2020-11-10 10x Raw Data Used In Paper (Paper V2)/LE-03 - Slide04_Section01 (Fig 3.b)/OCTAligned.tiff"  # @param {type:"string"}
    oct_input_image_path = os.path.join(image_directory, filename)
    # Load OCT image
    oct_image = cv2.imread(oct_input_image_path)
    oct_image = cv2.cvtColor(oct_image, cv2.COLOR_BGR2RGB)
    #is it sheered?
    right_column = oct_image.shape[1]-1
    if (oct_image[:,0,0] == 0).all() or (oct_image[:,right_column,0] == 0).all():
        continue
    # OCT image's pixel size
    microns_per_pixel_z = 1
    microns_per_pixel_x = 1

    # no need to crop - the current folder contains pre cropped images.
    # cropped = crop_oct(oct_image)

    #workaround: for some reason the images look close to the target shape, but not exactly.
    oct_image = cv2.resize(oct_image, [1024, 512], interpolation=cv2.INTER_AREA)

    #for good input points, we need the gel masked out.
    masked_gel_image = mask_gel_and_low_signal(oct_image)

    # run vh&e
    virtual_histology_image, _, o2h_input = oct2hist.run_network(oct_image,
                                                                                             microns_per_pixel_x=microns_per_pixel_x,
                                                                                             microns_per_pixel_z=microns_per_pixel_z)
    # mask
    input_points, input_labels = get_sam_input_points(masked_gel_image, virtual_histology_image)

    predictor.set_image(virtual_histology_image)
    # masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label,
    #                                           multimask_output=False, )
    # mask, score = masks[0], scores[0]
    plt.figure(figsize=(10, 10))

    mask = None
    for i in range(len(input_points)):
        first_i_input_points = input_points[:i]
        first_i_input_labels = input_labels[:i]
        if mask is not None:
            mask = cv2.resize(mask.astype(int), (256,256), interpolation = cv2.INTER_NEAREST )
            mask = mask.astype(bool)
            mask = mask[np.newaxis,:,:]
        masks, scores, logits = predictor.predict(point_coords=first_i_input_points, point_labels=first_i_input_labels,
                                                  multimask_output=False, mask_input = mask)
        mask, score = masks[0], scores[0]

        if i % 5 == 0:
            plt.imshow(virtual_histology_image)
            show_mask(mask, plt.gca())
            show_points(first_i_input_points, first_i_input_labels, plt.gca())
            plt.axis('off')
            plt.show()
            time.sleep(2)

    plt.imshow(virtual_histology_image)
    show_mask(mask, plt.gca())
    show_points(input_points, input_labels, plt.gca())
    plt.axis('off')
    plt.show()
    time.sleep(2)

# score
# @title Notebook Inputs { display-mode: "form" }
# @markdown Input Image Path
# Path to the OCT image
