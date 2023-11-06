import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
from dataclasses_json import dataclass_json
import supervision as sv
from supervision import Detections


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
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
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
    dataset = project.version(1).download(rf_dataset)
    DATA_SET_SUBDIRECTORY = "test"
    ANNOTATIONS_FILE_NAME = "_annotations.coco.json"
    IMAGES_DIRECTORY_PATH = os.path.join(dataset.location, DATA_SET_SUBDIRECTORY)
    ANNOTATIONS_FILE_PATH = os.path.join(dataset.location, DATA_SET_SUBDIRECTORY, ANNOTATIONS_FILE_NAME)
    print(IMAGES_DIRECTORY_PATH)
    print(ANNOTATIONS_FILE_PATH)


def visualize_sam_masking():
    boolean_masks = [
        masks2['segmentation']
        for masks2
        in sorted(masks2, key=lambda x: x['area'], reverse=True)
    ]

    sv.plot_images_grid(
        images=boolean_masks,
        grid_size=(len(masks), int(len(boolean_masks) / 4)),
        size=(16, 16)
    )
def sam_masking(inject_real_histology_and_segmentation = False):
    import sys
    sys.path.append("..")
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

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
