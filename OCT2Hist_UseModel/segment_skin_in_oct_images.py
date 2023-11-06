import utils.pix2pix as pix2pix
import cv2
import numpy as np

# Run this function to set up the Neural Network with Pre-trained segment skin network
def setup_network():
  g_path = "/Users/dannybarash/Library/CloudStorage/GoogleDrive-dannybarash7@gmail.com/Shared drives/Yolab - Current Projects/_Datasets/2020-11-10 10x OCT2Hist Model (Paper V2)/latest_net_G.pth"

  pix2pix.setup_network(g_path,"segment_skin")

# This function evaluates the neural network on input image
# Inputs:
#   oct_image - input oct image in cv format (256x256x3). Input image should be scanned with 10x lens and z-stacked
# Outputs:
#   mask - specifying for each pixel is it inside tissue (true) or outside tissue (false)
def run_network (oct_image):
  # Rescale
  original_height, original_width = oct_image.shape[:2]
  input_image = cv2.resize(oct_image, [256,256] , interpolation=cv2.INTER_AREA)
  
  # Run the neural net
  mask_image = pix2pix.run_network(input_image,"segment_skin", netG_flag="")

  # Rescale output image back
  mask_rescaled_image = cv2.resize(cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY), [original_width, original_height] , interpolation=cv2.INTER_AREA)

  # Convert the color image to grayscale and filter to bolean
  boolean_image = mask_rescaled_image > 127

  return boolean_image
