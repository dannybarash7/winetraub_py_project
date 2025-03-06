import os
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np

from OCT2Hist_UseModel.utils.crop import crop_oct_for_pix2pix, crop, crop_histology_around_com
from OCT2Hist_UseModel.utils.gray_level_rescale import gray_level_rescale, gray_level_rescale_v2
from OCT2Hist_UseModel.utils.masking import mask_gel_and_low_signal
from OCT2Hist_UseModel import oct2hist
from zero_shot_segmentation.consts import DOWNSAMPLE_SAM_INPUT, CROP_HISTOLOGY, GEL_BOTTOM_ROW, APPLY_MASKING, \
    SHRINK_BCC
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


def erode_mask(gel_mask, by = 5):
    # Convert the boolean array to an 8-bit integer array
    int_array = gel_mask.astype(np.uint8)
    kernel = np.ones((by, by), np.uint8)  # 3x3 kernel for dilation
    int_array = cv2.erode(int_array, kernel, iterations=1)
    # dilated_array = cv2.dilate(eroded_array, kernel, iterations=1)
    # Optionally, convert the result back to a boolean array
    return int_array.astype(bool)


def get_gel_mask_from_masked_image(masked_gel_image):
    masked_gel_image = masked_gel_image[:,:,0]
    row_indices = np.indices(masked_gel_image.shape)[0]
    h = masked_gel_image.shape[0]
    if GEL_BOTTOM_ROW is not None:
        top_30p = row_indices < GEL_BOTTOM_ROW
    else:
        top_30p = row_indices < h/3
    black_pixels = masked_gel_image == 0
    gel_mask = np.logical_and(top_30p, black_pixels)
    eroded_mask = erode_mask(gel_mask, by = 21)
    return eroded_mask


def top_half_bottom_most_black_row(virtual_histology_image):
    gray_hist = cv2.cvtColor(virtual_histology_image, cv2.COLOR_BGR2GRAY)
    # Convert image to grayscale (if all channels are 0, grayscale will also be 0)
    # Sum across each row and find where the sum is 0 (indicating all zeros in that row)
    h,w,c = virtual_histology_image.shape
    gray_hist[int(h/2):,0]=1 #prevent sum to be 0 at bottom half of the image
    row_sums = np.sum(gray_hist, axis=1)
    zero_rows = np.where(row_sums == 0)[0]
    if zero_rows.size > 0:
        topmost_zero_row = zero_rows[-1]
    else:
        topmost_zero_row = -1
    return topmost_zero_row


def crop_mask_x_percent_from_left(cropped_bcc_mask_true,x=10):
    coords = np.argwhere(cropped_bcc_mask_true)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    new_x_min = x_min+int(x/100*(x_max-x_min))
    cropped_bcc_mask_true[:,x_min:new_x_min] = False
    return cropped_bcc_mask_true

def crop_mask_x_percent(cropped_bcc_mask_true, percent=10):
    coords = np.argwhere(cropped_bcc_mask_true)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    w,h = x_max-x_min, y_max- y_min
    w_delta = int((percent / 100) * w)
    h_delta = int((percent / 100) * h)
    new_x_min = x_min +  w_delta
    new_x_max = x_max - w_delta
    new_y_min = y_min + h_delta
    new_y_max = y_max - h_delta
    cropped_bcc_mask_true = np.zeros_like(cropped_bcc_mask_true)
    cropped_bcc_mask_true[new_y_min:new_y_max,new_x_min:new_x_max] = True
    return cropped_bcc_mask_true

def crop_mask_to_non_black_values(cropped_bcc_mask_true, virtual_histology_image):
    gray_hist = cv2.cvtColor(virtual_histology_image, cv2.COLOR_BGR2GRAY)
    coords = np.argwhere(cropped_bcc_mask_true)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop image A according to the bounding box from mask B
    gray_hist = gray_hist[y_min:y_max + 1, x_min:x_max + 1]
    # Convert image to grayscale (if all channels are 0, grayscale will also be 0)
    # Sum across each row and find where the sum is 0 (indicating all zeros in that row)
    row_sums = np.sum(gray_hist, axis=1)

    # Find the topmost row where the sum is 0
    zero_rows = np.where(row_sums == 0)[0]
    updated_mask = deepcopy(cropped_bcc_mask_true)
    if zero_rows.size > 0:
        topmost_zero_row = zero_rows[0] + y_min    # assuming below tissue black aread
        updated_mask[topmost_zero_row:,:] = False
    # col_sums = np.sum(gray_hist, axis=0)
    # zero_cols = np.where(col_sums == 0)[0]
    # if zero_cols.size > 0:
    #     left_most_zero_column = zero_cols[0] + x_min   #assuming right of tissue black aread
    #     updated_mask[:,left_most_zero_column:] = False
    return updated_mask


def predict_oct(oct_image, filename, mask_true, weights_path, args, create_vhist = True, output_vhist_path = None,
                prompts = None, dont_care_mask = None, vhist_path = None, bcc_mask_true = None):
    # Load OCT image
    # oct_image = cv2.imread(oct_input_image_path)
    # filename = oct_input_image_path.split('/')[-1]
    # OCT image's pixel size
    microns_per_pixel_z = 1
    microns_per_pixel_x = 1
    # for good input points, we need the gel masked out.
    crop_args, cropped_dont_care_mask, cropped_histology_gt, cropped_oct_unscaled, scaled_cropped_oct_without_gel,cropped_bcc_mask_true = preprocess_oct(dont_care_mask, oct_image,
                                                                                                                                   mask_true, bcc_mask_true)
    

    if create_vhist:
        # run vh&e
        virtual_histology_image, _, _ = oct2hist.run_network(cropped_oct_unscaled,
                                                                     microns_per_pixel_x=microns_per_pixel_x,
                                                                     microns_per_pixel_z=microns_per_pixel_z, apply_masking=APPLY_MASKING)

        # else:
        #     print("DEBUG: using vhist path", vhist_path)
        #     virtual_histology_image = cv2.imread(vhist_path, cv2.IMREAD_UNCHANGED)
        #
        #     if output_vhist_path:
        #         cv2.imwrite(output_vhist_path, virtual_histology_image)
        #     #crop top black part
        #     # top_line = top_half_bottom_most_black_row(virtual_histology_image)
        #     # if top_line>0:
        #     #     # Remove the first 40 lines
        #     #     image_cropped = virtual_histology_image[top_line:, :]
        #     #     # Create a black padding (40 rows of zeros)
        #     #     black_padding = np.zeros((top_line, image_cropped.shape[1], image_cropped.shape[2]), dtype=np.uint8)
        #     #     # Add the black padding at the bottom
        #     #     virtual_histology_image = np.vstack((image_cropped, black_padding))
        #     # Split the path into directory, base filename, and extension
        #     directory, filename = os.path.split(output_vhist_path)
        #     name, ext = os.path.splitext(filename)
        #
        #     # Append "original" before the file extension
        #     new_filename = f"{name}_aligned.{ext}"
        #     # Join the directory with the new filename to get the full path
        #     new_file_path = os.path.join(directory, new_filename)
        #     if output_vhist_path:
        #         cv2.imwrite(new_file_path, virtual_histology_image)
        if output_vhist_path:
            cv2.imwrite(output_vhist_path, virtual_histology_image)
        if cropped_bcc_mask_true is not None and cropped_bcc_mask_true.any():
            cropped_bcc_mask_true = crop_mask_to_non_black_values(cropped_bcc_mask_true, virtual_histology_image)
        # cropped_bcc_mask_true = crop_mask_x_percent_from_left(cropped_bcc_mask_true, x=30)
        if SHRINK_BCC:
            cropped_bcc_mask_true = crop_mask_x_percent(cropped_bcc_mask_true, percent=10)

        #take the R channel
        # virtual_histology_image = cv2.cvtColor(virtual_histology_image,cv2.COLOR_BGR2RGB)



        #segment the epidermis
        segmentation, points_used, prompts = run_gui_segmentation(virtual_histology_image, weights_path, gt_mask = cropped_histology_gt, args = args, prompts = prompts, dont_care_mask = cropped_dont_care_mask, filename=filename)
        if bcc_mask_true is not None:
        #segment the bcc
            
            bcc_segmentation, points_used, prompts = run_gui_segmentation(virtual_histology_image, weights_path,
                                                                          gt_mask=cropped_bcc_mask_true, args=args,
                                                                          prompts=prompts,
                                                                          dont_care_mask=cropped_dont_care_mask)
            ###HACK
            segmentation = bcc_segmentation
            ###HACK
        else:
            bcc_segmentation = None

    else:
        segmentation, points_used, prompts = run_gui_segmentation(scaled_cropped_oct_without_gel, weights_path,
                                                                  gt_mask = cropped_histology_gt, args = args,
                                                                  prompts = prompts, dont_care_mask = cropped_dont_care_mask, filename=filename)
        if bcc_mask_true is not None:
            # segment the bcc

            bcc_segmentation, points_used, prompts = run_gui_segmentation(scaled_cropped_oct_without_gel, weights_path,
                                                                          gt_mask=cropped_bcc_mask_true, args=args,
                                                                          prompts=prompts,
                                                                          dont_care_mask=cropped_dont_care_mask)
        else:
            bcc_segmentation = None
        virtual_histology_image = None
    # bounding_rectangle = utils.bounding_rectangle(cropped_histology_gt)
    return segmentation, virtual_histology_image, cropped_histology_gt, cropped_oct_unscaled, points_used, mask_true, prompts, crop_args, scaled_cropped_oct_without_gel, bcc_segmentation,cropped_bcc_mask_true

def preprocess_histology(dont_care_mask, oct_image, warped_mask_true):
    com_yx = get_center_of_mass(warped_mask_true)
    com_xy = com_yx[1],com_yx[0]
    cropped_histology, crop_args = crop_histology_around_com(oct_image,com_xy)
    cropped_histology_gt = crop(warped_mask_true, **crop_args)
    cropped_dont_care_mask = crop(dont_care_mask, **crop_args)
    return crop_args, cropped_dont_care_mask, cropped_histology_gt, cropped_histology

def predict_histology(histology_image,filename, mask_true, weights_path, args, create_vhist = True, output_vhist_path = None, prompts = None, dont_care_mask = None):
    # Load OCT image
    # histology_image = cv2.imread(oct_input_image_path)
    # filename = oct_input_image_path.split('/')[-1]
    # Utility functions to handle file operations


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
    segmentation, points_used, prompts = run_gui_segmentation(cropped_histology, weights_path, gt_mask = cropped_histology_gt,
                                                              args = args, prompts = prompts,
                                                              dont_care_mask = cropped_dont_care_mask,filename=filename)
    virtual_histology_image = None
    # bounding_rectangle = utils.bounding_rectangle(cropped_histology_gt)
    return segmentation, virtual_histology_image, cropped_histology_gt, cropped_histology, points_used, warped_mask_true, prompts, crop_args


def preprocess_oct(dont_care_mask, oct_image, warped_mask_true,bcc_mask_true):
    rescaled = gray_level_rescale_v2(oct_image)
    tissue_image, low_signal_masked_image = mask_gel_and_low_signal(oct_image)
    gel_mask = get_gel_mask_from_masked_image(tissue_image)
    oct_without_gel = rescaled.copy()
    oct_without_gel[gel_mask != 0] = 0
    # y_tissue_top = get_y_min_of_tissue(tissue_image)
    # if y_tissue_top > TARGET_TISSUE_HEIGHT:
    #     #CONFIG
    #     delta = 80#y_tissue_top - TARGET_TISSUE_HEIGHT
    # else:
    #     delta = 0
    #CONFIG - should the cropped OCT be rescaled for vhist or not? oct_image/rescaled?

    cropped_oct_unscaled, crop_args = crop_oct_for_pix2pix(oct_image)
    cropped_histology_gt = crop(warped_mask_true, **crop_args)
    scaled_cropped_oct_without_gel = crop(oct_without_gel, **crop_args)
    cropped_dont_care_mask = crop(dont_care_mask, **crop_args)
    cropped_bcc_mask_true = crop(bcc_mask_true, **crop_args)
    return crop_args, cropped_dont_care_mask, cropped_histology_gt, cropped_oct_unscaled, scaled_cropped_oct_without_gel,cropped_bcc_mask_true

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