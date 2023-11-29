import numpy as np

# This function rescales OCT image to make sure we take advantage of the full dynamic range
# input is an OCT image (n by m by 3), where 0 means masked out / non existing value
def gray_level_rescale(image):
  # Find the first percentile of lowest intensities
  min_non_zero = np.percentile(image[np.nonzero(image)],1)

  # Scale the image between 1 and 255
  scaled_image = ((image - min_non_zero) / (255 - min_non_zero) * 254 + 1)

  # Set the values less than or equal to 0 back to 0
  scaled_image[scaled_image <= 0] = 0
  scaled_image = scaled_image.astype(np.uint8)
  return scaled_image
  
