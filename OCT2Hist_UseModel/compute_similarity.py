# This module computes similarity between two images for a given resolution

import cv2
import numpy as np
from skimage import color
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error as mse

# Compute similarity between im1 and im2 (load using openCV).
# We can add blur radius for gaussian bluring (pixels)
# Returns ssim, mse
def compute_similarity (im1, im2, blur_radius=0):
  # Function to calculate SSIM between two images
  def calculate_similarity(image1, image2):
      # Convert images to grayscale
      gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
      gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
  
      # Compute different similarities
      ssim1, _ = ssim(gray1, gray2, full=True)
      mse1 = mse(gray1, gray2)

      return ssim1, mse1
    
  # Blur the images
  blurred_im1 = blur_image(im1, blur_radius)
  blurred_im2 = blur_image(im2, blur_radius)

  # Compute simularity
  return calculate_similarity(blurred_im1,blurred_im2),

# Blur image using gauissian filter
def blur_image(image, blur_radius):
  if blur_radius>0:
      sigma = blur_radius
      filter_size = int(2 * np.ceil(2 * sigma) + 1) # the default filter size in Matlab
      filt_img = cv2.GaussianBlur(image, (filter_size, filter_size), sigma)
      return filt_img
  else:
    return image
