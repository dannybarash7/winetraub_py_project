import numpy as np
from PIL import Image, ImageDraw

from zero_shot_segmentation.consts import rf_api_key, rf_workspace, rf_project_name, \
    rf_dataset_type, version

def coco_mask_to_numpy(image_shape, coco_mask):
    """
    Convert COCO format segmentation mask to a NumPy array.

    Parameters:
    - image_shape: Tuple (m, n) representing the shape of the image.
    - coco_mask: List of coordinates [x1, y1, x2, y2, ..., xn, yn] in COCO format.

    Returns:
    - numpy_mask: NumPy array of shape (m, n) with True within the mask boundaries and False elsewhere.
    """
    # Create an image and draw the polygon defined by the COCO mask
    mask_image = Image.new("1", image_shape[::-1], 0)
    draw = ImageDraw.Draw(mask_image)
    draw.polygon(coco_mask, outline=1, fill=1)
    del draw

    # Convert the mask image to a NumPy array
    numpy_mask = np.array(mask_image, dtype=bool)

    return numpy_mask


def download_images_and_masks():
    from roboflow import Roboflow
    rf = Roboflow(api_key=rf_api_key)
    project = rf.workspace(rf_workspace).project(rf_project_name)
    dataset = project.version(version).download(rf_dataset_type, overwrite=False)
    return dataset
