# This util is a wrapper to the pix2pix algorithm and allows
# us to evaluate the model by providing an input image

import cv2
import numpy as np
import os
import shutil
import subprocess

# Temporary folder that we use to collect and organize files
# Base folder already contains pix2pix code from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/9f8f61e5a375c2e01c5187d093ce9c2409f409b0
current_folder = os.path.dirname(os.path.realpath(__file__))
one_above_folder = desired_folder = os.path.dirname(current_folder)
base_folder =  one_above_folder + "/pytorch_CycleGAN_and_pix2pix/"

# Private module to wrapp around using cmd
def _run_subprocess(cmd):
    result = subprocess.run([cmd], capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        print(f"{cmd}")
        print(f"Command failed with exit code {result.returncode}.")
        print("Error message:")
        print(result.stderr)
        raise RuntimeError("See error")

# Create folder if it doesn't exist
def _create_folder_if_doesnt_exist(folder_name):
  # Check if the folder exists
  if not os.path.exists(folder_name):
      # If it doesn't exist, create it
      os.mkdir(folder_name)
  else:
      print(f"Folder '{folder_name}' already exists.")

# Run this function to set up the Neural Network with Pre-trained generator network
# path_to_generaor_network - path to generator network (*.pth file), default value is the OCT2Hist network
# model_name - pick your favorate name for the model
def setup_network(
    path_to_generaor_network, model_name
    ):
    
    # Install dependencies
    _run_subprocess(f'pip install -r {base_folder}/requirements.txt')
    
    # Mount google drive (if not already mounted)
    # drive.mount('/content/drive/')

    # Copy model parameters to the correct location
    _create_folder_if_doesnt_exist(f'{base_folder}/checkpoints/')
    _create_folder_if_doesnt_exist(f'{base_folder}/checkpoints/{model_name}/')
    _run_subprocess(f'cp "{path_to_generaor_network}" {base_folder}/checkpoints/{model_name}/latest_net_G.pth')

# This function evaluates the neural network on input image
# Inputs:
#   im - input image (input domain, e.g. OCT) in cv format (256x256x3). Input image should be masked and cropped.
#   model_name - same name as you gave the model in setup_network step
#   netG_flag - specify --netG
# Outputs:
#   output image (in target domain, e.g. virtual histology) in cv format
def run_network (im, model_name, netG_flag="--netG resnet_9blocks"):
    
    # Input check
    if im.shape[:2] != (256, 256):
        raise ValueError("Image size must be 256x256 pixels to run model on.")

    # Pix2Pix implementation expects 256x512 image. Pad with zeros
    padded = np.zeros([256,512,3], np.uint8)
    padded[:,:256,:] = im[:,:,:]
    
    # Write input image in the right folder structure
    images_dir = f"{base_folder}/dataset/test"
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    os.makedirs(images_dir)
    im_input_path = os.path.join(images_dir,"im1.jpg")
    cv2.imwrite(im_input_path, padded)

    #from "../pytorch_CycleGAN_and_pix2pix" import test
    # Run pix2pix
    _run_subprocess(f'python {base_folder}/test.py {netG_flag} --dataroot "{base_folder}/dataset/"  --model pix2pix --name {model_name} --checkpoints_dir "{base_folder}/checkpoints" --results_dir "{base_folder}/results" --num_test 1000')

    # Load output image
    im_output_path = f"{base_folder}/results/{model_name}/test_latest/images/im1_fake_B.png"
    im_output = cv2.imread(im_output_path)
    im_output = cv2.cvtColor(im_output, cv2.COLOR_BGR2RGB)

    return(im_output)
    
