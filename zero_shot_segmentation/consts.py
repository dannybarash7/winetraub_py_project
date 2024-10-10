import os

MEDSAM = True
SAMMED_2D = False
SAM = False

rf_api_key = "R04BinsZcBZ6PsfKR2fP"
rf_workspace = "yolab-kmmfx"
rf_project_name = "bcc-egkye"#"paper_data" #"paper_data"
rf_dataset_type = "coco-segmentation"  # "png-mask-semantic"
version = 1

INTERACTIVE_POINT_PREDICTION = True
DOWNSAMPLE_SAM_INPUT = False
SEGMENT_TILES = True
class COLORS:
    GT = [8 / 255, 255 / 255, 128 / 255, 0.6]
    PREDICTED_EPIDERMISE_BLUE = [0 / 255, 128 / 255, 255 / 255, 0.6]
    PREDICTED_EPIDERMISE_ORANGE = [255 / 255, 128 / 255, 0 / 255, 0.6]
    DONT_CARE = [255 / 255, 0 / 255, 0 / 255, 0.6]

# rf_project_name = "2024.4.30_83f_st2_cheek_10x_1_r2"
# rf_dataset_type = "coco-segmentation"  # "png-mask-semantic"
# version = 2

#CONFIG
TARGET_TISSUE_HEIGHT = 50
ANNOTATED_DATA = True
CONST_BOX = [0, 160, 1000, 400]
NPOINTS_FOR_SEGMENTATION = 30
MASK_SCALE_FACTOR = 0.15
CROP_HISTOLOGY = True
RUN_FIVE_TIMES = False
# ROBOFLOW_ANNOT_DATASET_DIR = "/Users/dannybarash/Code/oct/medsam/zero_shot_segmentation_test_sam/2024.4.30_83F_ST2_Cheek_10x_1_R2-1_CE/test"
# ROBOFLOW_ANNOT_DATASET_DIR  = os.path.join(os.getcwd(), f"./{rf_project_name}-{version}/test")
ROBOFLOW_ANNOT_DATASET_DIR  = os.path.join(os.getcwd(), f"./BCC-{version}/test")