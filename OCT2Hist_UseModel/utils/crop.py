import numpy as np

# Crop part of the image, pad with nan if target size is bigger than input image
def crop(input_image, target_width=1024, target_height=512, x0=0, z0=0):
    # Get the dimensions of the input image
    input_height, input_width = input_image.shape[:2]
    if input_height == target_height and input_width == target_width:
        return input_image
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
    if input_image.dtype == bool:
        fill_value = False
    else:
        fill_value = 0
    cropped_image_padded = np.pad(cropped_image, pad_width, mode='constant', constant_values=fill_value)

    return cropped_image_padded

def find_crop_coords(input_image, y_center): #left top corner
    input_height, input_width = input_image.shape[:2]
    x0 = int(max(input_width/2 - 1024/2,0))
    z0 = int(max(y_center - 512/2,0))


    # x0 = 200  # @param {type:"slider", min:0, max:1000, step:10}
    # z0 = 110  # @param {type:"slider", min:0, max:1000, step:10}
    target_width = 1024
    target_height = 512
    return target_width, target_height, x0, z0

def crop_oct(input_image, y_center):
    target_width, target_height, x0, z0 = find_crop_coords(input_image, y_center)
    coords = {"target_width":target_width, "target_height":target_height,"x0":x0,"z0":z0}
    cropped_img = crop(input_image, **coords)
    return cropped_img , coords