import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from OCT2Hist_UseModel.utils.crop import crop_oct
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

    return warped_image

def warp_oct(oct_image):
    margin = 10
    height,width,_ = oct_image.shape
    first_row = oct_image[0, :, 0]
    non_zero_indices = np.nonzero(first_row)[0]
    x = [non_zero_indices[0]+margin,0] #0 stands for first row
    y = [non_zero_indices[-1]-margin,0]  #0 stands for first row
    last_row = oct_image[-1, :, 0]
    non_zero_indices = np.nonzero(last_row)[0]
    z = [non_zero_indices[0]+margin,height-1]
    w = [non_zero_indices[-1]-margin,height-1]
    source_points = np.float32([x,y,z,w])

    target_points = np.float32([[0, 0], [width,0], [0,height-1], [width-1,height-1]])
    return warp_image(oct_image, source_points, target_points)


def predict(oct_input_image_path, predictor, weights_path, vhist = True):
    # Load OCT image
    oct_image = cv2.imread(oct_input_image_path)
    oct_image = cv2.cvtColor(oct_image, cv2.COLOR_BGR2RGB)
    # is it sheered?
    right_column = oct_image.shape[1] - 1
    if (oct_image[:, 0, 0] == 0).all() or (oct_image[:, right_column, 0] == 0).all():
        oct_image = warp_oct(oct_image)
    # top glowing layer workaround:
    if os.path.basename(oct_input_image_path) == 'LF-01-Slide04_Section02-Fig-3-d-_jpeg.rf.686dda2850b99806206cb905623f33a7.jpg':
        oct_image = oct_image[200:,:,:]
    # OCT image's pixel size
    microns_per_pixel_z = 1
    microns_per_pixel_x = 1
    # for good input points, we need the gel masked out.
    rescaled = gray_level_rescale(oct_image)
    masked_gel_image = mask_gel_and_low_signal(oct_image)
    y_center = get_y_center_of_tissue(masked_gel_image)
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
        # mask
        # input_point, input_label = get_sam_input_points(masked_gel_image, virtual_histology_image)
        #
        # predictor.set_image(virtual_histology_image)
        # masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label,
        #                                          multimask_output=False, )
        segmentation = run_gui_segmentation(virtual_histology_image, weights_path)
    else:
        segmentation = run_gui_segmentation(cropped, weights_path)
        masked_gel_image = None

    return segmentation, masked_gel_image, crop_args


def get_y_center_of_tissue(oct_image):
    non_zero_coords = np.column_stack(np.where(oct_image > 0))
    center_y = np.mean(non_zero_coords[:, 0])
    return center_y