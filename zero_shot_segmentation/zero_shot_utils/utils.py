from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional

from dataclasses_json import dataclass_json
import supervision as sv
from supervision import Detections

from zero_shot_segmentation.consts import MASK_SCALE_FACTOR


def score_masking(masks2, segmentation_mask):
    masks = sorted(masks2, key=lambda x: x['area'], reverse=True)
    mask0 = masks[0]['segmentation']
    print(mask0.sum().sum())
    print(segmentation_mask.sum().sum())

    intersection = (segmentation_mask & mask0).sum().sum()
    print(f"intersection:{intersection}")
    union = (segmentation_mask | mask0).sum().sum()
    print(f"union:{union}")
    score = float(intersection / union)
    print(f"masking_score:{score}")
    return score




def visualize_masks(masks):
    boolean_masks = [
        masks['segmentation']
        for masks
        in sorted(masks, key=lambda x: x['area'], reverse=True)
    ]
    
    sv.plot_images_grid(
        images=boolean_masks,
        grid_size=(len(masks), int(len(boolean_masks) / 4)),
        size=(16, 16)
    )

def visualize_masks_on_img(im, masks, figsize):
    plt.figure(figsize=figsize)
    plt.imshow(im)
    show_anns(masks)
    plt.axis('off')
    plt.show()

def show_anns(anns):

  if len(anns) == 0:
    return
  sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
  ax = plt.gca()
  ax.set_autoscale_on(False)

  img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
  img[:,:,3] = 0
  for ann in sorted_anns:
      m = ann['segmentation']
      color_mask = np.concatenate([np.random.random(3), [0.35]])
      img[m] = color_mask
  ax.imshow(img)

@dataclass_json
@dataclass
class COCOCategory:
    id: int
    name: str
    supercategory: str


@dataclass_json
@dataclass
class COCOImage:
    id: int
    width: int
    height: int
    file_name: str
    license: int
    date_captured: str
    coco_url: Optional[str] = None
    flickr_url: Optional[str] = None


@dataclass_json
@dataclass
class COCOAnnotation:
    id: int
    image_id: int
    category_id: int
    segmentation: List[List[float]]
    area: float
    bbox: Tuple[float, float, float, float]
    iscrowd: int


@dataclass_json
@dataclass
class COCOLicense:
    id: int
    name: str
    url: str


@dataclass_json
@dataclass
class COCOJson:
    images: List[COCOImage]
    annotations: List[COCOAnnotation]
    categories: List[COCOCategory]
    licenses: List[COCOLicense]

def init_sam(model_type,sam_checkpoint ):
    import sys
    sys.path.append("..")
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator
def load_coco_json(json_file: str) -> COCOJson:
    import json

    with open(json_file, "r") as f:
        json_data = json.load(f)

    return COCOJson.from_dict(json_data)


class COCOJsonUtility:
    @staticmethod
    def get_annotations_by_image_id(coco_data: COCOJson, image_id: int) -> List[COCOAnnotation]:
        return [annotation for annotation in coco_data.annotations if annotation.image_id == image_id]

    @staticmethod
    def get_annotations_by_image_path(coco_data: COCOJson, image_path: str) -> Optional[List[COCOAnnotation]]:
        image = COCOJsonUtility.get_image_by_path(coco_data, image_path)
        if image:
            return COCOJsonUtility.get_annotations_by_image_id(coco_data, image.id)
        else:
            return None

    @staticmethod
    def get_image_by_path(coco_data: COCOJson, image_path: str) -> Optional[COCOImage]:
        for image in coco_data.images:
            if image.file_name == image_path:
                return image
        return None

    @staticmethod
    def annotations2detections(annotations: List[COCOAnnotation]) -> Detections:
        class_id, xyxy = [], []

        for annotation in annotations:
            x_min, y_min, width, height = annotation.bbox
            class_id.append(annotation.category_id)
            xyxy.append([
                x_min,
                y_min,
                x_min + width,
                y_min + height
            ])

        return Detections(
            xyxy=np.array(xyxy, dtype=int),
            class_id=np.array(class_id, dtype=int)
        )

def get_roboflow_data(dir):
    import os
    from roboflow import Roboflow
    os.chdir(dir)
    rf_dir = "/content/roboflow"
    rf_api_key = "R04BinsZcBZ6PsfKR2fP"
    rf_workspace = "yolab-kmmfx"
    rf_project = "zero-shot-oct"
    rf_dataset = "png-mask-semantic"
    rf = Roboflow(api_key=rf_api_key)
    project = rf.workspace(rf_workspace).project(rf_project)
    dataset = project.rf_dataset_version(1).download(rf_dataset)
    DATA_SET_SUBDIRECTORY = "test"
    ANNOTATIONS_FILE_NAME = "_annotations.coco.json"
    IMAGES_DIRECTORY_PATH = os.path.join(dataset.location, DATA_SET_SUBDIRECTORY)
    ANNOTATIONS_FILE_PATH = os.path.join(dataset.location, DATA_SET_SUBDIRECTORY, ANNOTATIONS_FILE_NAME)
    print(IMAGES_DIRECTORY_PATH)
    print(ANNOTATIONS_FILE_PATH)


def sam_masking(inject_real_histology_and_segmentation = False):
    import sys
    sys.path.append("..")
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    # sam
    points_per_side = 32
    pred_iou_thresh = 0.90
    stability_score_thresh = 0.95
    crop_n_layers = 1
    crop_n_points_downscale_factor = 2
    min_mask_region_area = 3000

    mask_generator_2 = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
    )

    if inject_real_histology_and_segmentation:
        # Load OCT image
        real_histology_image_path = "/content/rf_dir/Zero-shot-oct-1/train/Hist-1_png.rf.168d6c48c79ccf974a2a1ecac761d3f5.jpg"
        real_histology_image = cv2.imread(real_histology_image_path)
        real_histology_image = cv2.cvtColor(real_histology_image, cv2.COLOR_BGR2RGB)
        histology_image_resized = real_histology_image
        cropped = real_histology_image

    masks2 = mask_generator_2.generate(histology_image_resized)
    return masks2

def remove_fill(matrix):
    start_row, start_col = 1,1
    rows, cols = len(matrix), len(matrix[0])
    output = deepcopy(matrix)

    def is_valid(row, col):
        return (0 < row < rows-1) and (0 < col < cols-1)

    def flood_fill(row, col):
        if not is_valid(row, col):
            output[row][col] = matrix[row][col]
            return

        four_neighbours = matrix[row-1][col] & matrix[row][col-1] & matrix[row+1][col] & matrix[row][col + 1]
        two_neighbours_vert = False#matrix[row - 1][col] & matrix[row + 1][col]
        two_neighbours_horiz = False#matrix[row][col - 1] & matrix[row][col + 1]
        if four_neighbours or two_neighbours_vert or two_neighbours_horiz:
            output[row][col] = False  # Mark the current cell as visited

        # Recursively visit neighboring cells
        flood_fill(row + 1, col)
        flood_fill(row, col + 1)

    # Perform flood-fill starting from the specified coordinates
    flood_fill(start_row, start_col)
    return output

def expand_bounding_rectangle(array, rect_shape):
    """
        :param array: x1,y1,x2,y2 coordinates
        :param rect_shape: img shape (height, width) which bounds the enlarged rectangle.
        :returns: 2x bbox bounded by rect_shape
    """
    x1,y1,x2,y2 = array
    width = x2-x1
    height = y2 - y1
    center = (x1 + x2) / 2, (y1 + y2) / 2
    next_x1 = max(0,center[0]-width)
    next_x2 = min(rect_shape[1], center[0] + width)
    next_y1 = max(0, center[1] - height)
    next_y2 = min(rect_shape[0], center[1] + height)
    return np.array([next_x1,next_y1,next_x2,next_y2])


def bounding_rectangle(array):
    rows, cols = np.any(array, axis=1), np.any(array, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    # x1, x2 = np.where(cols)[0][[0, -1]]
    #expand by 20%
    x1, x2 = np.where(cols)[0][[0, -1]]
    y1 = y1 -5
    h = y2 - y1
    #0.2 CONFIG
    y2 = y2 + MASK_SCALE_FACTOR * h
    y2 = int(min(y2, array.shape[0]))
    # x1 = x1 + 60
    # x2 = x2 - 60
    # y1 = y1 + 20
    # y2 = y2 - 30
    return np.array([x1,y1,x2,y2])


def get_center_of_mass(boolean_array):
    # Find the indices where values are True
    true_indices = np.argwhere(boolean_array)
    com = np.average(true_indices, axis=0)
    com = (np.round(com)).astype(int)
    return com

def pad(image):
    height, width = image.shape[:2]
    OCT2HIST_HEIGHT = 512
    OCT2HIST_WIDTH = 1024
    # Calculate padding amounts
    pad_height = max(0, OCT2HIST_HEIGHT - height)
    pad_width = max(0, OCT2HIST_WIDTH - width)

    # Calculate padding for each side
    pad_top = 0
    pad_bottom = pad_height
    pad_left = 0
    pad_right = pad_width

    # Pad the image
    if len(image.shape) == 2:
        padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    else:
        padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')

    return padded_image


def calculate_iou(mask_true, mask_pred, class_id, dont_care_mask):
    # intersection = np.logical_and(mask_true == class_id, mask_pred == class_id)
    if dont_care_mask is not None:
        mask_pred = mask_pred & (~dont_care_mask)
    intersection = np.logical_and(mask_true, (mask_pred == class_id))
    union = np.logical_or(mask_true, mask_pred == class_id)
    true_count_gt = np.sum(mask_true)
    true_count_pred = np.sum(mask_pred)
    true_count_intersection = np.sum(intersection)
    iou = true_count_intersection / np.sum(union)
    dice = 2 * true_count_intersection / (true_count_gt + true_count_pred)
    return iou, dice


def calculate_iou_for_multiple_predictions(mask_true, mask_predictions, class_id, dont_care_mask):
    max_dice, max_iou = -1.0, -1.0
    best_mask = None
    for mask_pred in mask_predictions:
        iou, dice = calculate_iou(mask_true, mask_pred, class_id, dont_care_mask)
        if dice > max_dice:
            max_iou = iou
            max_dice = dice
            best_mask = mask_pred
    return max_iou, max_dice, best_mask

def extract_oct_base_name(filename, st):
    name = filename.replace(st, "")
    i = name.find("_A")
    return name[:i]

def single_or_multiple_predictions(mask_true, mask_predictions, class_id, dont_care_mask):
    if isinstance(mask_predictions, list):
        return calculate_iou_for_multiple_predictions(mask_true, mask_predictions, class_id, dont_care_mask)
    else:
        return calculate_iou(mask_true, mask_predictions, class_id, dont_care_mask)


def interpolate_masks(mask1, mask2, w):
    if mask1 is None or mask2 is None:
        return None
    """
    Interpolates between two 2D boolean masks and thresholds the result.

    Parameters:
    mask1 (numpy.ndarray): First 2D boolean mask.
    mask2 (numpy.ndarray): Second 2D boolean mask.
    w (float): Weight for interpolation (between 0 and 1).

    Returns:
    numpy.ndarray: A 2D boolean array, thresholded at 0.5.
    """

    # Convert boolean masks to float values [0, 1]
    mask1_float = mask1.astype(float)
    mask2_float = mask2.astype(float)

    # Interpolate with weights w and 1-w
    interpolated = w * mask1_float + (1 - w) * mask2_float

    # Threshold the result at 0.5
    result = interpolated >= 0.5

    return result

def make_mask_drawable(mask):
    mask = mask.astype(np.uint8)
    mask[mask == 1] = 255
    return mask


def extract_filename_prefix(filename):
    # Split the filename based on the dot ('.') and take the first part
    prefix = filename.split('.')[0]

    # Remove the "_jpg" part if it exists
    if prefix.endswith('_jpg'):
        prefix = prefix[:-4]
    if prefix.endswith('_png'):
        prefix = prefix[:-4]

    return prefix
