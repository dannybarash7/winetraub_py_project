import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from OCT2Hist_UseModel.utils.crop import crop_oct_for_pix2pix, crop
from OCT2Hist_UseModel.utils.gray_level_rescale import gray_level_rescale
from OCT2Hist_UseModel.utils.masking import get_sam_input_points, show_points, show_mask, mask_gel_and_low_signal
from OCT2Hist_UseModel import oct2hist
from zero_shot_segmentation.zero_shot_utils import utils
from zero_shot_segmentation.zero_shot_utils.run_sam_gui import run_gui_segmentation

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

def predict(oct_input_image_path, mask_true, weights_path, args, create_vhist = True, downsample = False, output_vhist_path = None):
    # Load OCT image
    oct_image = cv2.imread(oct_input_image_path)
    # is it sheered?
    # right_column = oct_image.shape[1] - 1
    # if is_trapezoid_image(oct_image) and mask_true is not None:
    #     oct_image, crop_coords = crop_oct_from_trapezoid(oct_image)
    #     # #TODO: check the warped mask true path...
    #     mask_true_uint8 = mask_true.astype(np.uint8) * 255
    #     warped_mask_true = mask_true_uint8[crop_coords[0]: crop_coords[1], crop_coords[2]:crop_coords[3]]
    #     # warped_mask_true = utils.pad(warped_mask_true)
    #     # warped_mask_true = cv2.warpPerspective(mask_true_uint8, affine_transform_matrix, (mask_true.shape[1], mask_true.shape[0]))
    #     warped_mask_true = (warped_mask_true > 0)
    # else:
    warped_mask_true = mask_true
    # OCT image's pixel size
    microns_per_pixel_z = 1
    microns_per_pixel_x = 1
    # for good input points, we need the gel masked out.
    rescaled = gray_level_rescale(oct_image)
    masked_gel_image = mask_gel_and_low_signal(oct_image)
    y_center = get_y_center_of_tissue(masked_gel_image)
    y_center = y_center * (2/3) #center of tissue should be around 2/3 height.
    # no need to crop - the current folder contains pre cropped images.
    cropped, crop_args = crop_oct_for_pix2pix(rescaled, y_center)
    cropped_histology_gt = crop(warped_mask_true, **crop_args)

    # Calculate the histogram
    # histogram = cv2.calcHist([cropped], [0], None, [256], [0, 256])

    # Plot the histogram
    # plt.plot(histogram)
    # plt.title('Grayscale Image Histogram')
    # plt.xlabel('Pixel Value')
    # plt.ylabel('Frequency')
    # plt.show()

    if create_vhist:

        # run vh&e
        virtual_histology_image, _, o2h_input = oct2hist.run_network(cropped,
                                                                     microns_per_pixel_x=microns_per_pixel_x,
                                                                     microns_per_pixel_z=microns_per_pixel_z)

        #take the R channel
        virtual_histology_image = cv2.cvtColor(virtual_histology_image,cv2.COLOR_BGR2RGB)

        if output_vhist_path:
            cv2.imwrite(output_vhist_path, virtual_histology_image)
        # virtual_histology_image = virtual_histology_image[:,:,0]
        virtual_histology_image_copy = virtual_histology_image.copy()
        if downsample:
            blurred_image = cv2.GaussianBlur(virtual_histology_image, (0, 0), 4)
            downsampled_image = cv2.resize(blurred_image, (0, 0), fx=0.25, fy=0.25)
            virtual_histology_image = downsampled_image
        # mask
        # input_point, input_label = get_sam_input_points(masked_gel_image, virtual_histology_image)
        #
        # predictor.set_image(virtual_histology_image)
        # masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label,
        #                                          multimask_output=False, )
        segmentation, points_used, prompts = run_gui_segmentation(virtual_histology_image, weights_path, gt_mask = cropped_histology_gt, args = args)
        if downsample:
            segmentation = cv2.resize(segmentation, (0, 0), fx=4, fy=4)

    else:
        segmentation, points_used, prompts = run_gui_segmentation(cropped, weights_path, gt_mask = cropped_histology_gt, args = args)
        virtual_histology_image_copy = None
    bounding_rectangle = utils.bounding_rectangle(cropped_histology_gt)
    return segmentation, virtual_histology_image_copy, cropped_histology_gt, cropped, points_used, warped_mask_true, prompts, bounding_rectangle


def get_y_center_of_tissue(oct_image):
    non_zero_coords = np.column_stack(np.where(oct_image > 0))
    center_y = np.mean(non_zero_coords[:, 0])
    return center_y