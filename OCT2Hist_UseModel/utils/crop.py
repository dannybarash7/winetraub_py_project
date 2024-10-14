import numpy as np

# Crop part of the image, pad with nan if target size is bigger than input image
def crop(input_image, target_width=1024, target_height=512, x0=0, z0=0):
    if input_image is None:
        return None
    # Get the dimensions of the input image
    input_height, input_width = input_image.shape[:2]
    if input_height == target_height and input_width == target_width:
        return input_image

    if input_image.dtype == bool:
        fill_value = False
    else:
        fill_value = 0

    # Pad top image if needed
    if z0 < 0:
        height_pad = (-z0, 0)
        width_pad = (0, 0)
        if len(input_image.shape) == 3:
            channels_pad = (0, 0)
            input_image = np.pad(input_image, (height_pad,width_pad,channels_pad), mode='constant', constant_values=fill_value)
        if len(input_image.shape) == 2:
            input_image = np.pad(input_image, (height_pad, width_pad), mode='constant',constant_values=fill_value)
    z0 = max(z0,0)

    # Calculate the cropping region
    x1 = min(input_width, x0 + target_width)
    z1 = min(input_height, z0 + target_height)

    # Crop the image
    cropped_image = input_image[z0:z1, x0:x1]

    # Add padding as needed to reach target width and height
    if len(input_image.shape) == 3:
        pad_width = ((0, max(0, target_height - (z1 - z0))), (0, max(0, target_width - (x1 - x0))), (0, 0))
    if len(input_image.shape) == 2:
        pad_width = ((0, max(0, target_height - (z1 - z0))), (0, max(0, target_width - (x1 - x0))))

    cropped_image_padded = np.pad(cropped_image, pad_width, mode='constant', constant_values=fill_value)

    return cropped_image_padded

def find_crop_coords(input_image, y_tissue_top, delta): #left top corner
    input_height, input_width = input_image.shape[:2]
    x0 = int(max(input_width/2 - 1024/2,0))
    z0 = int(y_tissue_top + delta - input_height / 2)
    # x0 = 200  # @param {type:"slider", min:0, max:1000, step:10}
    # z0 = 110  # @param {type:"slider", min:0, max:1000, step:10}
    target_width = 1024
    target_height = 512
    return target_width, target_height, x0, z0

def crop_oct_for_pix2pix(input_image):
    # target_width, target_height, x0, z0 = find_crop_coords(input_image, y_tissue_top, delta)
    target_width, target_height, x0, z0 = 1024,512,0,40
    coords = {"target_width":target_width, "target_height":target_height,"x0":x0,"z0":z0}
    cropped_img = crop(input_image, **coords)
    return cropped_img , coords

def crop_histology_around_com(input_image, com):
    target_width = 1024
    target_height = 512
    x0 = int(max(com[0] - target_width/2,0))
    z0 = int(max(com[1] - target_height/2,0))
    coords = {"target_width":target_width, "target_height":target_height,"x0":x0,"z0":z0}
    cropped_img = crop(input_image, **coords)
    return cropped_img , coords