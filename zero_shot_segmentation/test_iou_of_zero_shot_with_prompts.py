# -*- coding: utf-8 -*-
"""run_oct2hist.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/WinetraubLab/OCT2Hist-ModelInference/blob/main/run_oct2hist.ipynb

# Overview
Use this notebook to convert an OCT image to virtual histology.

To get started,
[open this notebook in colab](https://colab.research.google.com/github/WinetraubLab/OCT2Hist-ModelInference/blob/main/run_oct2hist.ipynb) and run.
"""
import argparse
import sys
import matplotlib.patches as patches
import numpy
import pandas
import pandas as pd
import torch
from matplotlib.patches import Patch

from zero_shot_segmentation.consts import MEDSAM, SAMMED_2D, SAM

sys.path.append("./OCT2Hist_UseModel/SAM_Med2D")
import segment_anything
from tqdm import tqdm
from OCT2Hist_UseModel.utils.masking import get_sam_input_points, show_points, show_mask, mask_gel_and_low_signal

from OCT2Hist_UseModel.utils.crop import crop
from zero_shot_segmentation.zero_shot_utils.predict_mask_on_oct_interactive import predict
sys.path.append('./OCT2Hist_UseModel')
# from google.colab import drive
sys.path.append('./zero_shot_segmentation')
import cv2
import matplotlib.pyplot as plt

import os

# Define the Roboflow project URL and API key
rf_api_key= "R04BinsZcBZ6PsfKR2fP"
rf_workspace= "yolab-kmmfx"
rf_project_name = "paper_data"
rf_dataset_type = "coco-segmentation" #"png-mask-semantic"
version = 2

if MEDSAM:
    CHECKPOINT_PATH = "/Users/dannybarash/Code/oct/medsam/MedSAM/work_dir/MedSAM/medsam_vit_b.pth"  # os.path.join("weights", "sam_vit_h_4b8939.pth")
if SAM:
    CHECKPOINT_PATH = "weights/sam_vit_h_4b8939.pth"  # os.path.join("weights", "sam_vit_h_4b8939.pth")
if SAMMED_2D:
    CHECKPOINT_PATH = None

roboflow_annot_dataset_dir = os.path.join(os.getcwd(),f"./paper_data-2/test")
#TODO: change this:
raw_oct_dataset_dir = "/Users/dannybarash/Library/CloudStorage/GoogleDrive-dannybarash7@gmail.com/Shared drives/Yolab - Current Projects/Yonatan/Hist Images/"
real_histology_dir = raw_oct_dataset_dir

#roboflow semantic classes
EPIDERMIS = True #mask values for epidermis mask

import numpy as np
from PIL import Image, ImageDraw

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

# Function to download images and masks from Roboflow
def download_images_and_masks(api_key, workspace, project_name, dataset_name, version):
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(version).download(rf_dataset_type, overwrite = False)
    return dataset

# Function to calculate Intersection over Union (IoU)
def calculate_iou(mask_true, mask_pred, class_id):
    #intersection = np.logical_and(mask_true == class_id, mask_pred == class_id)
    intersection = np.logical_and(mask_true, mask_pred == class_id)
    union = np.logical_or(mask_true, mask_pred == class_id)
    true_count_gt = np.sum(mask_true)
    true_count_pred =np.sum(mask_pred)
    true_count_intersection = np.sum(intersection)
    iou = true_count_intersection / np.sum(union)
    dice = 2*true_count_intersection / (true_count_gt+true_count_pred)
    return iou, dice

def calculate_iou_for_multiple_predictions(mask_true, mask_predictions, class_id):
    max_dice,max_iou = -1.0,-1.0
    best_mask = None
    for mask_pred in mask_predictions:
        iou,dice = calculate_iou(mask_true, mask_pred, class_id)
        if dice > max_dice:
            max_iou = iou
            max_dice = dice
            best_mask = mask_pred
    return max_iou, max_dice, best_mask

def single_or_multiple_predictions(mask_true, mask_predictions, class_id):
    if isinstance(mask_predictions, list):
        return calculate_iou_for_multiple_predictions(mask_true, mask_predictions, class_id)
    else:
        return calculate_iou(mask_true, mask_predictions, class_id)

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

    return prefix


def main(args):
    global df, roboflow_next_img
    # Download images and masks
    dataset = download_images_and_masks(rf_api_key, rf_workspace, rf_project_name, rf_dataset_type, version)
    # prepare model
    DEVICE = torch.device('mps')  # 'cpu'
    MODEL_TYPE = "vit_h"
    # Get the list of image files
    image_files = [f for f in os.listdir(roboflow_annot_dataset_dir) if f.endswith(".jpg")]
    image_files.sort()
    total_iou_vhist = {EPIDERMIS: 0}  # DERMIS:0 , # IOU for each class
    total_iou_oct = {EPIDERMIS: 0}
    total_dice_vhist = {EPIDERMIS: 0}
    total_dice_oct = {EPIDERMIS: 0}
    total_samples_vhist = 0
    total_samples_oct = 0
    path_to_annotations = os.path.join(roboflow_annot_dataset_dir, "_annotations.coco.json")
    from pylabel import importer
    dataset = importer.ImportCoco(path_to_annotations, path_to_images=roboflow_annot_dataset_dir, name="zero_shot_oct")
    visualize_input_gt = False
    # visualize_input_hist = False
    visualize_pred_vs_gt_vhist = False
    visualize_pred_vs_gt_oct = True
    visualize_pred_over_vhist = True
    visualize_input_vhist = True
    segment_real_hist = False
    skip_real_histology = False
    create_virtual_histology = True
    is_input_always_oct = True
    start_from_n = 1
    take_first_n_images = args.take_first_n if args.take_first_n > 0 else -1
    output_image_dir = args.output_dir
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    index_array = [extract_filename_prefix(file) for file in image_files]
    df = pd.DataFrame({
        "iou_vhist": numpy.nan,
        "nclicks_vhist": numpy.nan,
        "iou_oct": numpy.nan,  # Replace with your data for "iou oct"
        "nclicks_oct": numpy.nan,  # Replace with your data for "iou oct"
        "iou_hist": numpy.nan,
        "nclicks_hist": numpy.nan,
    }, index=index_array)
    i = 0

    def save_diff_image(oct_mask, cropped_histology_gt, path):
        oct_bool = oct_mask.astype(bool)
        hist_bool = cropped_histology_gt.astype(bool)
        diff = oct_bool ^ hist_bool
        diff = make_mask_drawable(diff)
        plt.figure(figsize=(5, 5))
        plt.imshow(diff)
        plt.axis('off')
        plt.savefig(f"{path}_diff.png")
        plt.close('all')

    if take_first_n_images > 0:
        image_files = image_files[:take_first_n_images]
    for oct_fname in tqdm(image_files):

        # if not extract_filename_prefix(oct_fname).startswith("LHC-31-Slide03_Section03_yp0_A"):
        #     continue
        # print("Skipping to LHC-31-Slide03_Section03_yp0_A... ")
        is_real_histology = oct_fname.find("_B_") != -1 or oct_fname.find("histology") != -1
        is_oct = oct_fname.find("oct") != -1 or is_input_always_oct
        # if is_real_histology and skip_real_histology:
        #     continue
        is_virtual_histology = oct_fname.find("vhist") != -1
        # if not is_virtual_histology:
        #     continue
        i += 1
        image_name = extract_filename_prefix(oct_fname)
        print(f"\nimage number {i}: {image_name}")
        if start_from_n > i:
            continue
        image_path = os.path.join(roboflow_annot_dataset_dir, oct_fname)
        roboflow_next_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # gt_by_histology = False
        # if gt_by_histology:
        #     is_oct = not is_virtual_histology
        #     real_histology_image_name = image_name.replace("_A","_B")
        #     real_histology_fname=dataset.df.img_filename[dataset.df.img_filename.str.startswith(real_histology_image_name)].values[0]
        #     real_histology_path = os.path.join(roboflow_annot_dataset_dir, real_histology_fname)
        #     real_hist_img = cv2.imread(real_histology_path, cv2.IMREAD_UNCHANGED)
        #
        #     coco_mask = dataset.df.ann_segmentation[dataset.df.img_filename == real_histology_fname].values[0][0]
        #     mask_true = coco_mask_to_numpy(roboflow_next_img.shape[:2], coco_mask)
        # else:
        coco_mask = dataset.df.ann_segmentation[dataset.df.img_filename == oct_fname].values[0][0]
        mask_true = coco_mask_to_numpy(roboflow_next_img.shape[:2], coco_mask)
        if visualize_input_gt:
            plt.figure(figsize=(5, 5))
            plt.imshow(roboflow_next_img)
            show_mask(mask_true, plt.gca(), alpha=0.3)
            plt.axis('off')
            plt.suptitle(f"Input oct and ground truth mask")
            plt.title(f"{image_name}")
            plt.savefig(f'{os.path.join(output_image_dir, image_name)}_input_gt.png')
            plt.close('all')
        # if visualize_input_hist:
        #     plt.figure(figsize=(5, 5))
        #     plt.imshow(real_hist_img)
        #     show_mask(mask_true, plt.gca(), alpha=0.3)
        #     plt.axis('off')
        #     plt.suptitle(f"Input real histology and ground truth mask")
        #     plt.title(f"{image_name}")
        #     plt.savefig(f'{os.path.join(output_image_dir, image_name)}_input_hist.png')
        #     plt.close('all')

        # oct
        if is_oct:
            print("OCT segmentation")
            oct_mask, _, cropped_histology_gt, cropped_oct_image, n_points_used, warped_mask_true, prompts  = predict(image_path, mask_true,
                                                                              args=args,
                                                                              weights_path=CHECKPOINT_PATH,
                                                                              create_vhist=False)

            path = f'{os.path.join(output_image_dir, image_name)}_cropped_oct_image.png'
            # save image to disk
            cv2.imwrite(path, cropped_oct_image)
            # Calculate IoU for each class# DERMIS
            mask_true = cropped_histology_gt
            if warped_mask_true is None or warped_mask_true.sum().sum() == 0:
                print(f"Could not segment OCT image {image_path}.")
            else:
                epidermis_iou_oct, dice, best_mask = single_or_multiple_predictions(mask_true, oct_mask, EPIDERMIS)
                print(f"OCT iou: {epidermis_iou_oct}.")
                print(f"OCT dice: {dice}.")
                total_iou_oct[EPIDERMIS] += epidermis_iou_oct
                total_dice_oct[EPIDERMIS] += dice
                df.loc[image_name, "iou_oct"] = epidermis_iou_oct
                df.loc[image_name, "dice_oct"] = dice
                df.loc[image_name, "nclicks_oct"] = n_points_used

                if visualize_pred_vs_gt_oct:
                    visualize_prediction(best_mask, cropped_histology_gt, cropped_oct_image, dice, image_name,
                                         output_image_dir, save_diff_image, prompts, ext = "oct_pred")

                total_samples_oct += 1

        if is_real_histology:
            # histology segmentation
            print("histology segmentation")

            if segment_real_hist:
                histology_mask, _, cropped_histology_gt, cropped_oct_image, n_points_used, warped_mask_true, prompts  = predict(image_path, mask_true,
                                                                                        args=args,
                                                                                        weights_path=CHECKPOINT_PATH,
                                                                                        create_vhist=False)
                if warped_mask_true is None or warped_mask_true.sum().sum() == 0:
                    print(f"Could not segment {image_path}.")
                    continue
                warped_mask_true[warped_mask_true == 1] = True
                warped_mask_true[warped_mask_true == 0] = False
                # cropped_histology_image = crop(real_hist_img, **crop_args)
                # path = f'{os.path.join(output_image_dir, image_name)}_cropped_histology.png'
                # cv2.imwrite(path, cropped_histology_image)
                # cropped_histology_gt = crop(warped_mask_true, **crop_args)
                epidermis_iou_real_hist, dice = calculate_iou(mask_true, histology_mask, EPIDERMIS)
                df.loc[image_name, "iou_hist"] = epidermis_iou_real_hist
                df.loc[image_name, "dice_hist"] = dice
                df.loc[image_name, "nclicks_hist"] = n_points_used
                print(f"real histology iou: {epidermis_iou_real_hist}.")
                print(f"real histology dice: {dice}.")
                plt.figure(figsize=(5, 5))
                plt.imshow(roboflow_next_img)
                c1 = show_mask(histology_mask, plt.gca())
                c2 = show_mask(mask_true, plt.gca(), secondcolor=True, alpha=0.2)
                plt.axis('off')
                plt.suptitle(f"Real histology segmentation: iou {epidermis_iou_real_hist:.2f}")
                plt.title(f"{image_name}")
                legend_elements = [
                    Patch(color=c1, alpha=1, label='Yours'),
                    Patch(color=c2, alpha=1, label='GT'),
                ]
                plt.legend(handles=legend_elements)

                fpath = f'{os.path.join(output_image_dir, image_name)}_pred_real_hist'
                plt.savefig(f'{fpath}.png')
                save_diff_image(histology_mask, mask_true, fpath)
                plt.close()

                plt.figure(figsize=(5, 5))
                plt.imshow(roboflow_next_img)
                show_mask(mask_true, plt.gca(), alpha=0.6)
                plt.axis('off')
                plt.suptitle(f"Input real histology and predicted mask, iou {epidermis_iou_real_hist:.2f}")
                plt.title(f"name {image_name}")
                plt.savefig(f'{os.path.join(output_image_dir, image_name)}_pred_hist.png')
                plt.close()
        if is_virtual_histology or create_virtual_histology:
            # v. histology segmentation
            print("virtual histology segmentation")
            path = f'{os.path.join(output_image_dir, image_name)}_cropped_vhist_image.png'
            cropped_vhist_mask, cropped_vhist, cropped_vhist_mask_true, cropped_oct_image, n_points_used, warped_vhist_mask_true, prompts  = predict(image_path,
                                                                                                          mask_true,
                                                                                                          args = args,
                                                                                                          weights_path=CHECKPOINT_PATH,
                                                                                                          create_vhist=create_virtual_histology,
                                                                                                          output_vhist_path=path)
            # cropped_vhist_mask_true = crop(warped_vhist_mask_true, **crop_args)
            if is_virtual_histology:
                cropped_vhist = roboflow_next_img
            if visualize_input_vhist:
                plt.figure(figsize=(5, 5))
                plt.imshow(cropped_vhist)
                show_mask(cropped_vhist_mask_true, plt.gca(), alpha=0.6)
                plt.axis('off')
                plt.suptitle(f"Generated vhist and ground truth mask")
                plt.title(f"name {image_name}")
                plt.savefig(f'{os.path.join(output_image_dir, image_name)}_input_vhist.png')
                plt.close()

            if len(cropped_vhist_mask) == 0:
                print(f"Could not segment {image_path}.")
                continue
            # cropped_vhist_mask[cropped_vhist_mask == 1] = True
            # cropped_vhist_mask[cropped_vhist_mask == 0] = False
            # cropped_oct_image = crop(roboflow_next_img, **crop_args)
            epidermis_iou_vhist, dice, best_mask = single_or_multiple_predictions(mask_true, cropped_vhist_mask,
                                                                                  EPIDERMIS)
            if best_mask is None:
                print(f"Could not calculate iou for {image_path}.")
                continue
            print(f"v. histology iou: {epidermis_iou_vhist}.")
            print(f"v. histology dice: {dice}.")
            df.loc[image_name, "iou_vhist"] = epidermis_iou_vhist
            df.loc[image_name, "dice_vhist"] = dice
            df.loc[image_name, "nclicks_vhist"] = n_points_used
            total_iou_vhist[EPIDERMIS] += epidermis_iou_vhist
            total_dice_vhist[EPIDERMIS] += dice
            total_samples_vhist += 1

            if visualize_pred_over_vhist:
                visualize_prediction(best_mask, mask_true, cropped_vhist, dice, image_name,
                                     output_image_dir, save_diff_image, prompts, ext = "vhist_pred")
                # plt.figure(figsize=(5, 5))
                # plt.imshow(cropped_vhist)
                # c1 = show_mask(best_mask, plt.gca())
                # c2 = show_mask(mask_true, plt.gca(), secondcolor=True, alpha=0.6)
                # plt.axis('off')
                # plt.suptitle(f"vhist segmentation: iou {epidermis_iou_vhist:.2f}")
                # plt.title(f"{image_name}")
                # # Add a legend
                # legend_elements = [
                #     Patch(color=c1, alpha=1, label='Yours'),
                #     Patch(color=c2, alpha=1, label='GT'),
                # ]
                # plt.legend(handles=legend_elements)
                # fpath = f'{os.path.join(output_image_dir, image_name)}_vhist_pred'
                # plt.savefig(f'{fpath}.png')
                # save_diff_image(best_mask, mask_true, fpath)
                # plt.close()

            # if visualize_pred_vs_gt_vhist:
            #     plt.figure(figsize=(5, 5))
            #     plt.imshow(cropped_oct_image, cmap = "gray")
            #     c1 = show_mask(cropped_vhist_mask, plt.gca())
            #     c2 = show_mask(mask_true, plt.gca(), random_color=True, alpha = 0.2)
            #     plt.axis('off')
            #     plt.suptitle(f"oct and vhist segmentation: iou {epidermis_iou_vhist:.2f}, {n_points_used} clicks")
            #     plt.title(f"{image_name}")
            #     # Add a legend
            #     legend_elements = [
            #         Patch(color=c1, alpha=1, label='Yours'),
            #         Patch(color=c2, alpha=1, label='GT'),
            #     ]
            #     plt.legend(handles=legend_elements)
            #     plt.savefig(f'{os.path.join(output_image_dir, image_name)}_oct_pred_with_vhist.png')
            #     plt.close()

        df.to_csv(os.path.join(output_image_dir, 'iou_scores.csv'), index=True)
    average_iou = total_iou_vhist[
                      EPIDERMIS] / total_samples_vhist  # sum all ious divided by (number of images * number of classes).
    print(f"Average IoU with virtual histology: {average_iou}")
    average_iou_oct = total_iou_oct[EPIDERMIS] / total_samples_oct
    print(f"Average IoU without virtual histology: {average_iou_oct}")
    average_dice = total_dice_vhist[
                       EPIDERMIS] / total_samples_vhist  # sum all dices divided by (number of images * number of classes).
    print(f"Average dice with virtual histology: {average_dice}")
    average_dice_oct = total_dice_oct[EPIDERMIS] / total_samples_oct
    print(f"Average dice without virtual histology: {average_dice_oct}")
    import numpy as np
    from scipy.stats import ttest_ind
    # df["dice_hist"].values
    # df["dice_vhist"].values
    # Generate two random arrays of floats
    # array1 = np.random.rand(100)
    # array2 = np.random.rand(100)
    array1 = df["dice_oct"].values
    array2 = df["dice_vhist"].values
    # Perform a two-sample t-test
    t_statistic, p_value = ttest_ind(array1, array2, nan_policy="omit", equal_var=False,)
    # Print the results
    print(f'T-statistic: {t_statistic}')
    print(f'P-value: {p_value}')
    # Interpret the results
    alpha = 0.05  # significance level
    if p_value < alpha:
        print('Reject the null hypothesis: There is a significant difference between the two groups.')
    else:
        print('Fail to reject the null hypothesis: There is no significant difference between the two groups.')

    str_to_save = (f'Average IoU with virtual histology: {average_iou}\n'
                   f'Average IoU without virtual histology: {average_iou_oct}'
                   f'Average dice with virtual histology: {average_dice}\n'
                   f'Average dice without virtual histology: {average_dice_oct}\n'
                   f'T-statistic: {t_statistic}, P-value: {p_value}, alpha: {alpha}, p_value < alpha: {p_value < alpha}')
    file_path = os.path.join(output_image_dir, 'p_value.txt')
    with open(file_path, 'w+') as file:
        file.write(str_to_save)
def visualize_prediction(best_mask, cropped_histology_gt, cropped_oct_image, dice, image_name, output_image_dir,
                         save_diff_image, prompts, ext):
    plt.figure(figsize=(5, 5))
    plt.imshow(cropped_oct_image, cmap="gray")
    c1 = show_mask(best_mask, plt.gca())
    c2 = show_mask(cropped_histology_gt, plt.gca(), secondcolor=True, outline=True)
    # c2 = show_mask(cropped_histology_gt, plt.gca(), random_color=True, alpha=0.2)
    plt.axis('off')
    # plt.suptitle(f"oct segmentation w/o vhist: iou {epidermis_iou_oct:.2f}")

    text_to_display = f"dice {dice:.2f}"
    plt.text(0.02, 0.9, text_to_display, color='white', fontsize=12, transform=plt.gca().transAxes)
    if args.point:
        add_pts,remove_pts = prompts["add"], prompts["remove"]
        #overlay points
        plt.scatter(remove_pts[:, 1], remove_pts[:, 0], color='red', marker='o', s=10)
        plt.scatter(add_pts[:, 1], add_pts[:, 0], color='lightgreen', marker='+', s=15)
    if args.box:
        #overlay box
        rectangle_coords = prompts['box']
        rectangle = patches.Rectangle((rectangle_coords[0], rectangle_coords[1]),
                                      rectangle_coords[2] - rectangle_coords[0],
                                      rectangle_coords[3] - rectangle_coords[1], linewidth=1, edgecolor='yellow',
                                      facecolor='none')
        plt.gca().add_patch(rectangle)
    # plt.title(f"{image_name}")
    # legend_elements = [
    #     Patch(color=c1, alpha=1, label='Yours'),
    #     Patch(color=c2, alpha=1, label='GT'),
    # ]
    # plt.legend(handles=legend_elements)
    fpath = f'{os.path.join(output_image_dir, image_name)}_{ext}'
    plt.savefig(f'{fpath}.png', bbox_inches='tight', pad_inches=0)
    save_diff_image(best_mask, cropped_histology_gt, fpath)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process point, box, and grid arguments.")
    group = parser.add_mutually_exclusive_group()
    parser.add_argument("--output_dir", help="Specify output directory, e.g. './images/point_prediction' ")
    parser.add_argument("--take_first_n", help="take first n images", default=-1, type=int)
    parser.add_argument("--npoints", help="number_of_prediction_points", default=10, type=int)
    group.add_argument("--point", action="store_true", help="Specify a point.")
    group.add_argument("--box", action="store_true", help="Specify a box.")
    group.add_argument("--grid", action="store_true", help="Specify a grid.")

    args = parser.parse_args()
    # if args.point:
    #     process_point()
    #
    # elif args.box:
    #     process_box()
    #
    # elif args.grid:
    #     process_grid()
    if not args.point and not args.box and not args.grid:
        print("Please specify one of --point, --box, or --grid.")
    elif not args.output_dir:
        print("Please specify output dir.")
    else:
        main(args)

