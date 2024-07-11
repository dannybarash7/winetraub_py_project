MEDSAM = False
SAMMED_2D = True
SAM = False

rf_api_key = "R04BinsZcBZ6PsfKR2fP"
rf_workspace = "yolab-kmmfx"
rf_project_name = "paper_data"
rf_dataset_type = "coco-segmentation"  # "png-mask-semantic"
version = 6

NPOINTS_FOR_SEGMENTATION = 20
DOWNSAMPLE_SAM_INPUT = True
class COLORS:
    GT = [8 / 255, 255 / 255, 128 / 255, 0.6]
    PREDICTED_EPIDERMISE_BLUE = [0 / 255, 128 / 255, 255 / 255, 0.6]
    PREDICTED_EPIDERMISE_ORANGE = [255 / 255, 128 / 255, 0 / 255, 0.6]
    DONT_CARE = [255 / 255, 0 / 255, 0 / 255, 0.6]
