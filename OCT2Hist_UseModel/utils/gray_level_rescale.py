import numpy as np

from zero_shot_segmentation.consts import MIN_SIGNAL_LEVEL_PERCENTILE


# This function rescales OCT image to make sure we take advantage of the full dynamic range
# input is an OCT image (n by m by 3), where 0 means masked out / non existing value
def gray_level_rescale_v2(image):
  # Find the first percentile of lowest intensities
  #15 for volume? 5 for normal? #CONFIG
  min_non_zero = np.percentile(image[np.nonzero(image)],MIN_SIGNAL_LEVEL_PERCENTILE)

  # Scale the image between 1 and 255
  scaled_image = ((image - min_non_zero) / (255 - min_non_zero) * 254 + 1)

  # Set the values less than or equal to 0 back to 0
  scaled_image[scaled_image <= 0] = 0
  scaled_image = scaled_image.astype(np.uint8)
  return scaled_image


def gray_level_rescale(image):
  # Find the smallest non-zero value in the image
  min_non_zero = np.min(image[np.nonzero(image)])

  # Scale the image between 1 and 255
  scaled_image = ((image - min_non_zero) / (255 - min_non_zero) * 254 + 1).astype(np.uint8)

  # Set the values less than or equal to 0 back to 0
  scaled_image[scaled_image <= 0] = 0

  return scaled_image
