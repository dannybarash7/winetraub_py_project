
import cv2
import matplotlib.pyplot as plt

from OCT2Hist_UseModel.utils.masking import get_sam_input_points, show_points, show_mask, mask_gel_and_low_signal
from OCT2Hist_UseModel import oct2hist

def predict(oct_input_image_path, predictor):
    # Load OCT image
    oct_image = cv2.imread(oct_input_image_path)
    oct_image = cv2.cvtColor(oct_image, cv2.COLOR_BGR2RGB)
    # is it sheered?
    right_column = oct_image.shape[1] - 1
    if (oct_image[:, 0, 0] == 0).all() or (oct_image[:, right_column, 0] == 0).all():
        print(f"{oct_input_image_path} is sheered, I can only segment full rectangular shapes.")
        return None,None
    # OCT image's pixel size
    microns_per_pixel_z = 1
    microns_per_pixel_x = 1

    # no need to crop - the current folder contains pre cropped images.
    # cropped = crop_oct(oct_image)

    # workaround: for some reason the images look close to the target shape, but not exactly.
    oct_image = cv2.resize(oct_image, [1024, 512], interpolation=cv2.INTER_AREA)

    # for good input points, we need the gel masked out.
    masked_gel_image = mask_gel_and_low_signal(oct_image)

    # run vh&e
    virtual_histology_image, _, o2h_input = oct2hist.run_network(oct_image,
                                                                 microns_per_pixel_x=microns_per_pixel_x,
                                                                 microns_per_pixel_z=microns_per_pixel_z)
    # mask
    input_point, input_label = get_sam_input_points(masked_gel_image, virtual_histology_image)

    predictor.set_image(virtual_histology_image)
    masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label,
                                              multimask_output=False, )

    return masks, masked_gel_image