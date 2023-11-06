import numpy as np

# This function rescales OCT image to make sure we take advantage of the full dynamic range
# input is an OCT image (n by m by 3), where 0 means masked out / non existing value
def gray_level_rescale(image):
  # Find the smallest non-zero value in the image
  min_non_zero = np.min(image[np.nonzero(image)])

  # Scale the image between 1 and 255
  scaled_image = ((image - min_non_zero) / (255 - min_non_zero) * 254 + 1).astype(np.uint8)

  # Set the values less than or equal to 0 back to 0
  scaled_image[scaled_image <= 0] = 0

  return scaled_image
  
