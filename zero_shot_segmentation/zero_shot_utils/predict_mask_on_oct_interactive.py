
import cv2
import matplotlib.pyplot as plt

from OCT2Hist_UseModel.utils.crop import crop_oct
from OCT2Hist_UseModel.utils.masking import get_sam_input_points, show_points, show_mask, mask_gel_and_low_signal
from OCT2Hist_UseModel import oct2hist
from zero_shot_segmentation.zero_shot_utils.run_sam_gui import run_gui_segmentation


def predict(oct_input_image_path, predictor, weights_path, vhist = True):
    # Load OCT image
    oct_image = cv2.imread(oct_input_image_path)
    oct_image = cv2.cvtColor(oct_image, cv2.COLOR_BGR2RGB)
    # is it sheered?
    right_column = oct_image.shape[1] - 1
    if (oct_image[:, 0, 0] == 0).all() or (oct_image[:, right_column, 0] == 0).all():
        print(f"{oct_input_image_path} is sheered, I can only segment full rectangular shapes.")
        return None,None, None
    # OCT image's pixel size
    microns_per_pixel_z = 1
    microns_per_pixel_x = 1

    # no need to crop - the current folder contains pre cropped images.
    cropped, crop_args =  crop_oct(oct_image)

    # workaround: for some reason the images look close to the target shape, but not exactly.
    #oct_image = cv2.resize(cropped, [1024, 512], interpolation=cv2.INTER_AREA)
    oct_image = cropped

    if vhist:
        # for good input points, we need the gel masked out.
        masked_gel_image = mask_gel_and_low_signal(oct_image)

        # run vh&e
        virtual_histology_image, _, o2h_input = oct2hist.run_network(oct_image,
                                                                     microns_per_pixel_x=microns_per_pixel_x,
                                                                     microns_per_pixel_z=microns_per_pixel_z)
        # mask
        # input_point, input_label = get_sam_input_points(masked_gel_image, virtual_histology_image)
        #
        # predictor.set_image(virtual_histology_image)
        # masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label,
        #                                          multimask_output=False, )
        segmentation = run_gui_segmentation(virtual_histology_image, weights_path)
    else:
        segmentation = run_gui_segmentation(oct_image, weights_path)
        masked_gel_image = None

    return segmentation, masked_gel_image, crop_args