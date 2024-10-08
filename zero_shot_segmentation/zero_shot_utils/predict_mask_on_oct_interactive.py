import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from OCT2Hist_UseModel.utils.crop import crop_oct_for_pix2pix, crop, crop_histology_around_com
from OCT2Hist_UseModel.utils.gray_level_rescale import gray_level_rescale, gray_level_rescale_v2
from OCT2Hist_UseModel.utils.masking import mask_gel_and_low_signal
from OCT2Hist_UseModel import oct2hist
from zero_shot_segmentation.consts import DOWNSAMPLE_SAM_INPUT, TARGET_TISSUE_HEIGHT, CROP_HISTOLOGY
from zero_shot_segmentation.zero_shot_utils.run_sam_gui import run_gui_segmentation
from zero_shot_segmentation.zero_shot_utils.utils import get_center_of_mass


def warp_image(source_image, source_points, target_points):
    # Convert the input points to NumPy arrays
    src_pts = np.float32(source_points)
    dst_pts = np.float32(target_points)

    # Calculate the affine transformation matrix
    affine_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the affine transformation to the source image
    warped_image = cv2.warpPerspective(source_image, affine_matrix, (source_image.shape[1], source_image.shape[0]))

    return warped_image, affine_matrix


def calculate_bottom_corners(height, top_left, top_right, middle_left, middle_right):
    # Calculate the slopes of the left and right sides
    left_slope = (top_left[0] - middle_left[0]) / (top_left[1] - middle_left[1])
    right_slope = (top_right[0] - middle_right[0]) / (top_right[1] - middle_right[1])

    # Calculate bottom left and bottom right points
    bottom_left_y = height - 1
    bottom_right_y = height - 1
    bottom_left_x = np.round(middle_left[0] + (bottom_left_y - middle_left[1]) * left_slope).astype(int)
    bottom_right_x =  np.round(middle_right[0] + (bottom_right_y - middle_right[1]) * right_slope).astype(int)


    return (bottom_left_x, bottom_left_y), (bottom_right_x, bottom_right_y)




def crop_oct_from_trapezoid(oct_image):

    height,width,_ = oct_image.shape
    mid_row = int(height/2)
    first_row = oct_image[0, :, 0]
    non_zero_indices = np.nonzero(first_row)[0]
    top_left = [non_zero_indices[0],0] #0 stands for first row
    top_right = [non_zero_indices[-1],0]  #0 stands for first row
    last_row = oct_image[mid_row, :, 0]
    non_zero_indices = np.nonzero(last_row)[0]
    middle_left = [non_zero_indices[0],mid_row]
    middle_right = [non_zero_indices[-1],mid_row]
    # source_points = np.float32([top_left,top_right,middle_left,middle_right])

    (bottom_left_x, bottom_left_y), (bottom_right_x, bottom_right_y) = calculate_bottom_corners(height, top_left, top_right, middle_left, middle_right)
    left_border_x =max(top_left[0],bottom_left_x)
    right_border_x = min(top_right[0], bottom_right_x)
    #pad right_border to width 1024
    right_border_x = max(right_border_x,  left_border_x + 1024)
    # pad bottom to height 512
    bottom_border_y = max(bottom_left_y, top_left[1]+512)
    top_border_y = top_left[1]
    crop_coords = top_border_y, bottom_border_y, left_border_x, right_border_x
    cropped_image = oct_image[crop_coords[0]: crop_coords[1], crop_coords[2]:crop_coords[3]]
    # cropped_image = utils.pad(cropped_image)
    return cropped_image,crop_coords


def is_trapezoid_image(oct_image):
    margin = 10
    height, width, _ = oct_image.shape
    first_row = oct_image[0, :, 0]
    top_row_first_non_zero_index = np.nonzero(first_row)[0][0]
    mid_row = int(height / 2)
    mid_row = oct_image[mid_row, :, 0]
    mid_row_first_non_zero_index = np.nonzero(mid_row)[0][0]
    if top_row_first_non_zero_index > margin or mid_row_first_non_zero_index > margin:
        return True


def raise_mask(gel_mask, by = 5):
    # Convert the boolean array to an 8-bit integer array
    int_array = gel_mask.astype(np.uint8)

    # Define the dilation kernel
    kernel = np.ones((by, by), np.uint8)  # 3x3 kernel for dilation

    # Apply the cv2.dilate function
    dilated_array = cv2.erode(int_array, kernel, iterations=1)

    # Optionally, convert the result back to a boolean array
    return dilated_array.astype(bool)


def get_gel_mask_from_masked_image(masked_gel_image):
    masked_gel_image = masked_gel_image[:,:,0]
    row_indices = np.indices(masked_gel_image.shape)[0]
    h = masked_gel_image.shape[0]
    above_middle_image = row_indices < h/2
    black_pixels = masked_gel_image == 0
    gel_mask = np.logical_and(above_middle_image, black_pixels)
    raise_gel_mask = raise_mask(gel_mask, by = 21)
    return raise_gel_mask


def predict_oct(oct_input_image_path, mask_true, weights_path, args, create_vhist = True, output_vhist_path = None, prompts = None, dont_care_mask = None):
    # Load OCT image
    oct_image = cv2.imread(oct_input_image_path)
    warped_mask_true = mask_true
    # OCT image's pixel size
    microns_per_pixel_z = 1
    microns_per_pixel_x = 1
    # for good input points, we need the gel masked out.
    crop_args, cropped_dont_care_mask, cropped_histology_gt, cropped_oct_unscaled, scaled_cropped_oct_without_gel = preprocess_oct(dont_care_mask, oct_image,
                                                                                          warped_mask_true)

    if create_vhist:

        # run vh&e

        virtual_histology_image, _, o2h_input = oct2hist.run_network(cropped_oct_unscaled,
                                                                     microns_per_pixel_x=microns_per_pixel_x,
                                                                     microns_per_pixel_z=microns_per_pixel_z)
        #take the R channel
        # virtual_histology_image = cv2.cvtColor(virtual_histology_image,cv2.COLOR_BGR2RGB)

        if output_vhist_path:
            cv2.imwrite(output_vhist_path, virtual_histology_image)

        if DOWNSAMPLE_SAM_INPUT:
            virtual_histology_image_copy = virtual_histology_image.copy()
            cropped_histology_gt_copy = cropped_histology_gt.copy()
            blurred_image = cv2.GaussianBlur(virtual_histology_image, (0, 0), 4)
            downsampled_image = cv2.resize(blurred_image, None, fx=0.25, fy=0.25)

            downscaled_img = cv2.resize(cropped_histology_gt.astype('float32'), None, fx=1 / 4,
                                        fy=1 / 4, interpolation=cv2.INTER_NEAREST)

            # Convert back to boolean image
            cropped_histology_gt = downscaled_img.astype('bool')

            # downscaled_img = cv2.resize(binary_img, None, fx=1/downscale_factor, fy=1/downscale_factor, interpolation=cv2.INTER_NEAREST)
            virtual_histology_image = downsampled_image

        segmentation, points_used, prompts = run_gui_segmentation(virtual_histology_image, weights_path, gt_mask = cropped_histology_gt, args = args, prompts = prompts, dont_care_mask = cropped_dont_care_mask)
        if DOWNSAMPLE_SAM_INPUT:
            assert(len(segmentation) == 1)
            segmentation = cv2.resize(segmentation[0].astype('float32'), (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            segmentation = [segmentation.astype('bool')]
            cropped_histology_gt = cropped_histology_gt_copy
            virtual_histology_image = virtual_histology_image_copy
            prompts["box"] = prompts["box"] * 4
    else:
        segmentation, points_used, prompts = run_gui_segmentation(scaled_cropped_oct_without_gel, weights_path, gt_mask = cropped_histology_gt, args = args, prompts = prompts, dont_care_mask = cropped_dont_care_mask)
        virtual_histology_image = None
    # bounding_rectangle = utils.bounding_rectangle(cropped_histology_gt)
    return segmentation, virtual_histology_image, cropped_histology_gt, cropped_oct_unscaled, points_used, warped_mask_true, prompts, crop_args, scaled_cropped_oct_without_gel

def preprocess_histology(dont_care_mask, oct_image, warped_mask_true):
    com_yx = get_center_of_mass(warped_mask_true)
    com_xy = com_yx[1],com_yx[0]
    cropped_histology, crop_args = crop_histology_around_com(oct_image,com_xy)
    cropped_histology_gt = crop(warped_mask_true, **crop_args)
    cropped_dont_care_mask = crop(dont_care_mask, **crop_args)
    return crop_args, cropped_dont_care_mask, cropped_histology_gt, cropped_histology


def predict_histology(oct_input_image_path, mask_true, weights_path, args, create_vhist = True, output_vhist_path = None, prompts = None, dont_care_mask = None):
    # Load OCT image
    histology_image = cv2.imread(oct_input_image_path)
    warped_mask_true = mask_true
    # OCT image's pixel size
    microns_per_pixel_z = 1
    microns_per_pixel_x = 1
    # for good input points, we need the gel masked out.
    if CROP_HISTOLOGY:
        crop_args, cropped_dont_care_mask, cropped_histology_gt, cropped_histology = preprocess_histology(dont_care_mask, histology_image,
                                                                                              warped_mask_true)
    else:
        crop_args = {"target_width": 1024, "target_height": 512, "x0": 0, "z0": 0}
        cropped_dont_care_mask, cropped_histology_gt, cropped_histology = dont_care_mask, warped_mask_true, histology_image
    segmentation, points_used, prompts = run_gui_segmentation(cropped_histology, weights_path, gt_mask = cropped_histology_gt, args = args, prompts = prompts, dont_care_mask = cropped_dont_care_mask)
    virtual_histology_image = None
    # bounding_rectangle = utils.bounding_rectangle(cropped_histology_gt)
    return segmentation, virtual_histology_image, cropped_histology_gt, cropped_histology, points_used, warped_mask_true, prompts, crop_args


def preprocess_oct(dont_care_mask, oct_image, warped_mask_true):
    rescaled = gray_level_rescale_v2(oct_image)
    tissue_image, low_signal_masked_image = mask_gel_and_low_signal(oct_image)
    gel_mask = get_gel_mask_from_masked_image(tissue_image)
    oct_without_gel = rescaled.copy()
    oct_without_gel[gel_mask != 0] = 0
    y_tissue_top = get_y_min_of_tissue(tissue_image)
    if y_tissue_top > TARGET_TISSUE_HEIGHT:
        #CONFIG
        delta = 80#y_tissue_top - TARGET_TISSUE_HEIGHT
    else:
        delta = 0
    #CONFIG - should the cropped OCT be rescaled for vhist or not? oct_image/rescaled?
    cropped_oct_unscaled, crop_args = crop_oct_for_pix2pix(oct_image, y_tissue_top, delta)
    cropped_histology_gt = crop(warped_mask_true, **crop_args)
    scaled_cropped_oct_without_gel = crop(oct_without_gel, **crop_args)
    cropped_dont_care_mask = crop(dont_care_mask, **crop_args)


    return crop_args, cropped_dont_care_mask, cropped_histology_gt, cropped_oct_unscaled, scaled_cropped_oct_without_gel

def get_y_center_of_tissue(oct_image):
    non_zero_coords = np.column_stack(np.where(oct_image > 0))
    center_y = np.mean(non_zero_coords[:, 0])
    return center_y

def get_y_min_of_tissue(oct_image):
    if len(oct_image.shape) == 3:
        oct_image = oct_image[:,:,0]
    top_non_black_pixels_y_values = np.argmax(oct_image > 0, axis=0)
    center_y = np.median(top_non_black_pixels_y_values)
    return center_y