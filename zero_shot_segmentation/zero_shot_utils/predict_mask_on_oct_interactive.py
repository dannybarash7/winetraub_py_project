import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from OCT2Hist_UseModel.utils.crop import crop_oct, crop
from OCT2Hist_UseModel.utils.gray_level_rescale import gray_level_rescale
from OCT2Hist_UseModel.utils.masking import get_sam_input_points, show_points, show_mask, mask_gel_and_low_signal
from OCT2Hist_UseModel import oct2hist
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

def warp_oct(oct_image):
    margin = 10
    height,width,_ = oct_image.shape
    mid_row = int(height/2)
    first_row = oct_image[0, :, 0]
    non_zero_indices = np.nonzero(first_row)[0]
    x = [non_zero_indices[0]+margin,0] #0 stands for first row
    y = [non_zero_indices[-1]-margin,0]  #0 stands for first row
    last_row = oct_image[mid_row, :, 0]
    non_zero_indices = np.nonzero(last_row)[0]
    z = [non_zero_indices[0]+margin,mid_row]
    w = [non_zero_indices[-1]-margin,mid_row]
    source_points = np.float32([x,y,z,w])

    target_points = np.float32([[0, 0], [width,0], [0,mid_row-1], [width-1,mid_row-1]])
    return warp_image(oct_image, source_points, target_points)


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

def predict(oct_input_image_path, mask_true, weights_path, vhist = True, downsample = False):
    # Load OCT image
    oct_image = cv2.imread(oct_input_image_path)
    oct_image = cv2.cvtColor(oct_image, cv2.COLOR_BGR2RGB)
    # is it sheered?
    right_column = oct_image.shape[1] - 1
    if is_trapezoid_image(oct_image) and mask_true is not None:
        oct_image, affine_transform_matrix = warp_oct(oct_image)
        #TODO: check the warped mask true path...
        mask_true_uint8 = mask_true.astype(np.uint8) * 255
        warped_mask_true = cv2.warpPerspective(mask_true_uint8, affine_transform_matrix, (mask_true.shape[1], mask_true.shape[0]))
        warped_mask_true = (warped_mask_true > 0)
    else:
        warped_mask_true = mask_true
    # OCT image's pixel size
    microns_per_pixel_z = 1
    microns_per_pixel_x = 1
    # for good input points, we need the gel masked out.
    rescaled = gray_level_rescale(oct_image)
    masked_gel_image = mask_gel_and_low_signal(oct_image)
    y_center = get_y_center_of_tissue(masked_gel_image)
    y_center = y_center * (2/3)
    # no need to crop - the current folder contains pre cropped images.
    cropped, crop_args =  crop_oct(rescaled, y_center)

    # Calculate the histogram
    # histogram = cv2.calcHist([cropped], [0], None, [256], [0, 256])

    # Plot the histogram
    # plt.plot(histogram)
    # plt.title('Grayscale Image Histogram')
    # plt.xlabel('Pixel Value')
    # plt.ylabel('Frequency')
    # plt.show()

    if vhist:

        # run vh&e
        virtual_histology_image, _, o2h_input = oct2hist.run_network(cropped,
                                                                     microns_per_pixel_x=microns_per_pixel_x,
                                                                     microns_per_pixel_z=microns_per_pixel_z)
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
        segmentation, points_used = run_gui_segmentation(virtual_histology_image, weights_path)
        if downsample:
            segmentation = cv2.resize(segmentation, (0, 0), fx=4, fy=4)

    else:
        segmentation, points_used = run_gui_segmentation(cropped, weights_path)
        virtual_histology_image_copy = None

    return segmentation, virtual_histology_image_copy, crop_args, points_used, warped_mask_true


def get_y_center_of_tissue(oct_image):
    non_zero_coords = np.column_stack(np.where(oct_image > 0))
    center_y = np.mean(non_zero_coords[:, 0])
    return center_y