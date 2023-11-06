from OCT2Hist_UseModel.utils.show_images import *
from OCT2Hist_UseModel.utils.masking import mask_image, mask_image_gel
from OCT2Hist_UseModel.utils.gray_level_rescale import gray_level_rescale
from OCT2Hist_UseModel.utils import pix2pix as pix2pix
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Run this function to set up the Neural Network with Pre-trained oct2hist generator network
def setup_network():
  pix2pix.setup_network("/Users/dannybarash/Code/oct/zero_shot_segmentation_test_sam/OCT2Hist_UseModel/pytorch_CycleGAN_and_pix2pix/checkpoints/segment_skin/latest_net_G.pth","oct2hist")

# This function evaluates the neural network on input image
# Inputs:
#   oct_image - input oct image in cv format (256x256x3). Input image should be scanned with 10x lens and z-stacked
#   microns_per_pixel_x - how many microns is each pixel on x direction (lateral direction). This is determined by B-Scan parameters, not the lens.
#   microns_per_pixel_z - how many microns is each pixel on z direction (axial direction). This is determined by spectrumeter width not light source FWHM.
# Preprocessing configuration. Set this parameters to false if you would like to skip them
#   apply_masking - should we perform the mask step?
#   min_signal_threshold - By default this is NaN, set to numeric value if you would like to apply a min threshold for masking rather than use algorithm.
#   apply_gray_level_scaling - should we rescale gray level to take full advantage of dynamic range?
#   appy_resolution_matching - should we match resolution to the trained images?
# Outputs:
#   output image (in target domain, e.g. virtual histology) in cv format
#   masked_image - if apply_masking=true, otherwise it will be identical to im 
#   network_input_image - the image that is loaded to the network
def run_network(oct_image,
                microns_per_pixel_x=1,
                microns_per_pixel_z=1,
                apply_masking=True,
                min_signal_threshold=np.nan,
                apply_gray_level_scaling=True,
                appy_resolution_matching=True,
                ):
  if apply_gray_level_scaling:
    oct_image = gray_level_rescale(oct_image)
  else:
    oct_image = oct_image

  # Mask
  if apply_masking:
    masked_image, *_ = mask_image(oct_image, min_signal_threshold=min_signal_threshold)
  else:
    masked_image = oct_image

  # Apply resolution matching
  original_height, original_width = masked_image.shape[:2]
  if appy_resolution_matching:
    # Compute compression ratio
    target_width = original_width * microns_per_pixel_x // 4  # Target resolution is 4 microns per pixel on x axis. We use // to round to integer
    target_height = original_height * microns_per_pixel_z // 2  # Target resolution is 2 microns per pixel on z axis. We use // to round to integer

    if target_width != 256 or target_height != 256:
      raise ValueError(
        f"OCT2Hist works on images which have total size of 1024 microns by 512 microns (x,z). Input oct_image has size of {original_width * microns_per_pixel_x} by {original_height * microns_per_pixel_z} microns. Please crop or pad image")

    # Apply the resolution change
    o2h_input = cv2.resize(masked_image, [target_height, target_width], interpolation=cv2.INTER_AREA)
  else:
    o2h_input = masked_image

  # Run the neural net
  virtual_histology_image = pix2pix.run_network(o2h_input, "oct2hist")

  # Post process, return image to original size
  virtual_histology_image_resized = cv2.resize(virtual_histology_image, [original_width, original_height],
                                               interpolation=cv2.INTER_AREA)

  return virtual_histology_image_resized, masked_image, o2h_input
