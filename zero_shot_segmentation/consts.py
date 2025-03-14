import os

MEDSAM = True
SAMMED_2D = False
SAM = False
SAM2 = False
#
rf_api_key = "R04BinsZcBZ6PsfKR2fP"
rf_workspace = "yolab-kmmfx"
rf_project_name = "paper_data" #"paper_data"
rf_dir_name = rf_project_name
rf_dataset_type = "coco-segmentation"  # "png-mask-semantic"
version = 7
segmentation_class = "epidermis"


# rf_api_key = "R04BinsZcBZ6PsfKR2fP"
# rf_workspace = "yolab-kmmfx"
# rf_project_name = "paper_data_histology" #"paper_data"
# rf_dataset_type = "coco-segmentation"  # "png-mask-semantic"
# version = 2

# rf_api_key = "R04BinsZcBZ6PsfKR2fP"
# rf_workspace = "yolab-kmmfx"
# rf_project_name = "40x-ljgu8" #"paper_data"
# rf_dir_name = "40x" #"paper_data"
# rf_dataset_type = "coco"  # "png-mask-semantic"
# segmentation_class = "epidermis"
# version = 1

#eyes 1: 500 images
# rf_api_key = "R04BinsZcBZ6PsfKR2fP"
# rf_workspace = "yolab-kmmfx"
# rf_project_name = "layers_detection-j1waj" #"paper_data"
# rf_dir_name = "layers_detection" #"paper_data"
# rf_dataset_type = "coco"  # "png-mask-semantic"
# segmentation_class = "Layers - v17 2024-08-19 8-57am"
# version = 1

# eyes 2: 4500 images
# rf_api_key = "R04BinsZcBZ6PsfKR2fP"
# rf_workspace = "yolab-kmmfx"
# rf_project_name = "oct-semantic-qyrkd" #"paper_data"
# rf_dir_name = "oct-semantic-" #"paper_data"
# rf_dataset_type = "coco-segmentation"  # "png-mask-semantic"
# segmentation_class = "IS-OS"# "RPE"# "OPL"# "ILM"#
# version = 1

INTERACTIVE_POINT_PREDICTION = False
DOWNSAMPLE_SAM_INPUT = False
SEGMENT_TILES = False
APPLY_MASKING = False
SHRINK_BCC = False
DEFAULT_NPOINTS = 1
class COLORS:
    GT = [8 / 255, 255 / 255, 128 / 255, 0.6]
    PREDICTED_EPIDERMISE_BLUE = [0 / 255, 128 / 255, 255 / 255, 0.6]
    PREDICTED_EPIDERMISE_ORANGE = [255 / 255, 128 / 255, 0 / 255, 0.6]
    DONT_CARE = [255 / 255, 0 / 255, 0 / 255, 0.6]

# rf_project_name = "2024.4.30_83f_st2_cheek_10x_1_r2"
# rf_dataset_type = "coco-segmentation"  # "png-mask-semantic"
# version = 2

#CONFIG
MIN_SIGNAL_FACTOR = 0.2 # factor*(high-low) will determine the min signal, deep tissue threshold.
MIN_SIGNAL_LEVEL_PERCENTILE = 5 # related - what is the signal percentile that decides what is noise.
TARGET_TISSUE_HEIGHT = 50
GEL_BOTTOM_ROW = 100 #can be NAN
ANNOTATED_DATA = True
CONST_BOX = [0, 160, 1000, 400]
NPOINTS_FOR_SEGMENTATION = 30
MASK_SCALE_FACTOR = 0.15
CROP_HISTOLOGY = True
RUN_FIVE_TIMES = False
# ROBOFLOW_ANNOT_DATASET_DIR = "/Users/dannybarash/Code/oct/medsam/zero_shot_segmentation_test_sam/2024.4.30_83F_ST2_Cheek_10x_1_R2-1_CE/test"
ROBOFLOW_ANNOT_DATASET_DIR  = os.path.join(os.getcwd(), f"./{rf_dir_name}-{version}/test")
# ROBOFLOW_ANNOT_DATASET_DIR  = os.path.join(os.getcwd(), f"./BCC-{version}/test")