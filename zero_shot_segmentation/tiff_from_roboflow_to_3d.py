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
import os
import pickle
import shutil
import sys

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy
import pandas
import pandas as pd
from tqdm import tqdm

from OCT2Hist_UseModel.utils.crop import crop
from OCT2Hist_UseModel.utils.masking import show_mask
from zero_shot_segmentation.consts import MEDSAM, SAMMED_2D, SAM, version, COLORS, ANNOTATED_DATA, rf_project_name
from zero_shot_segmentation.zero_shot_utils.ds_utils import coco_mask_to_numpy, download_images_and_masks

sys.path.append("./OCT2Hist_UseModel/SAM_Med2D")
from zero_shot_segmentation.zero_shot_utils.predict_mask_on_oct_interactive import predict_oct
from zero_shot_segmentation.zero_shot_utils.utils import single_or_multiple_predictions, extract_filename_prefix, \
    bounding_rectangle

sys.path.append('./OCT2Hist_UseModel')
sys.path.append('./zero_shot_segmentation')

# Define the Roboflow project URL and API key

# Flags
visualize_input_gt = False


# visualize_input_hist = False
visualize_pred_vs_gt_vhist = True
visualize_pred_vs_gt_oct = True
visualize_pred_over_vhist = True
visualize_input_vhist = True

create_virtual_histology = True
segment_real_hist = False
skip_existing_images =True
#None or filename
single_image_to_segment = None
patient_to_skip = None# ["LG-63", "LG-73", "LHC-36"]
#/Users/dannybarash/Code/oct/paper_code/3d_segmentation/rescaled_images/frame_0020_cropped_oct_image.png
# roboflow_annot_dataset_dir = os.path.join("/Users/dannybarash/Code/oct/paper_code/3d_segmentation/rescaled_images/")
roboflow_annot_dataset_dir = os.path.join("/Users/dannybarash/Code/oct/medsam/zero_shot_segmentation_test_sam/2024.4.30_83F_ST2_Cheek_10x_1_R2-1_CE/test")
# /Users/dannybarash/Code/oct/medsam/zero_shot_segmentation_test_sam/Reconstruction6_EnhancedContrast-1/test
raw_oct_dataset_dir = "/Users/dannybarash/Library/CloudStorage/GoogleDrive-dannybarash7@gmail.com/Shared drives/Yolab - Current Projects/Yonatan/Hist Images/"

if MEDSAM:
    CHECKPOINT_PATH = "/Users/dannybarash/Code/oct/medsam/MedSAM/work_dir/MedSAM/medsam_vit_b.pth"  # os.path.join("weights", "sam_vit_h_4b8939.pth")
if SAM:
    CHECKPOINT_PATH = "weights/sam_vit_h_4b8939.pth"  # os.path.join("weights", "sam_vit_h_4b8939.pth")
if SAMMED_2D:
    CHECKPOINT_PATH = None

# roboflow semantic classes
EPIDERMIS = True  # mask values for epidermis mask are simply True, where the foreground is False.

def segment_histology(image_path, epidermis_mask, image_name, dont_care_mask, prompts):
    global total_iou_histology, total_dice_histology
    print("histology segmentation")

    histology_mask, _, epidermis_mask, cropped_histology_image, n_points_used, warped_mask_true, prompts, crop_args = predict_oct(
        image_path, epidermis_mask, args=args, weights_path=CHECKPOINT_PATH, create_vhist=False, prompts=prompts)

    dont_care_mask = crop(dont_care_mask, **crop_args)
    path = f'{os.path.join(output_image_dir, image_name)}_cropped_histology_image.png'
    # save image to disk
    cv2.imwrite(path, cropped_histology_image)
    # Calculate IoU for each class# DERMIS
    if warped_mask_true is None or warped_mask_true.sum().sum() == 0:
        print(f"Could not segment histology image {image_path}.")
    else:
        epidermis_iou_real_hist, dice, best_mask = single_or_multiple_predictions(epidermis_mask, histology_mask,
                                                                                  EPIDERMIS,
                                                                                  dont_care_mask=dont_care_mask)
        df.loc[image_name, "iou_histology"] = epidermis_iou_real_hist
        df.loc[image_name, "dice_histology"] = dice

        print(f"real histology iou: {epidermis_iou_real_hist}.")
        print(f"real histology dice: {dice}.")
        if visualize_pred_vs_gt_oct:
            visualize_prediction(best_mask, epidermis_mask, dont_care_mask, cropped_histology_image, dice, image_name, output_image_dir,
                                 prompts, ext="histology_pred")
        total_iou_histology[EPIDERMIS] += epidermis_iou_real_hist
        total_dice_histology[EPIDERMIS] += dice
    # plt.figure(figsize=(5, 5))
    # roboflow_next_img = cv2.cvtColor(roboflow_next_img, cv2.COLOR_BGR2RGB)
    # plt.imshow(roboflow_next_img)
    # roboflow_next_img = cv2.cvtColor(roboflow_next_img, cv2.COLOR_BGR2RGB)
    # c1 = show_mask(histology_mask, plt.gca())
    # c2 = show_mask(mask_true, plt.gca(), secondcolor=True, alpha=0.2)
    # plt.axis('off')
    # plt.suptitle(f"Real histology segmentation: iou {epidermis_iou_real_hist:.2f}")
    # plt.title(f"{image_name}")
    # legend_elements = [
    #     Patch(color=c1, alpha=1, label='Yours'),
    #     Patch(color=c2, alpha=1, label='GT'),
    # ]
    # plt.legend(handles=legend_elements)
    #
    # fpath = f'{os.path.join(output_image_dir, image_name)}_pred_real_hist'
    # plt.savefig(f'{fpath}.png')
    # # save_diff_image(histology_mask, mask_true, fpath)
    # plt.close()
    #
    # plt.figure(figsize=(5, 5))
    # roboflow_next_img = cv2.cvtColor(roboflow_next_img, cv2.COLOR_BGR2RGB)
    # plt.imshow(roboflow_next_img)
    # roboflow_next_img = cv2.cvtColor(roboflow_next_img, cv2.COLOR_BGR2RGB)
    #
    # show_mask(mask_true, plt.gca(), alpha=0.6)
    # plt.axis('off')
    # plt.suptitle(f"Input real histology and predicted mask, iou {epidermis_iou_real_hist:.2f}")
    # plt.title(f"name {image_name}")
    # plt.savefig(f'{os.path.join(output_image_dir, image_name)}_pred_hist.png')
    # plt.close()

def segment_oct(image_path, epidermis_mask, image_name, dont_care_mask):
    global output_image_dir, total_iou_vhist, total_dice_vhist
    print("OCT segmentation")
    oct_mask, _, cropped_histology_gt, cropped_oct_image, n_points_used, warped_mask_true, prompts, crop_args = predict_oct(
        image_path, epidermis_mask, args=args, weights_path=CHECKPOINT_PATH, create_vhist=False, dont_care_mask = dont_care_mask)

    fpath = f'{os.path.join(output_image_dir, image_name)}_predicted_mask_oct.npy'
    with open(fpath, 'wb+') as f:
        numpy.save(f,oct_mask[0]) #a = numpy.load(fpath)

    crop_args_path = f'{os.path.join(output_image_dir, image_name)}_oct_crop_args.pickle'
    with open(crop_args_path, 'wb') as file:
        pickle.dump(crop_args, file)
    dont_care_mask = crop(dont_care_mask, **crop_args)
    path = f'{os.path.join(output_image_dir, image_name)}_cropped_oct_image.png'
    # save image to disk
    cv2.imwrite(path, cropped_oct_image)
    # Calculate IoU for each class# DERMIS
    epidermis_mask = cropped_histology_gt
    if warped_mask_true is None or warped_mask_true.sum().sum() == 0:
        print(f"Could not segment OCT image {image_path}.")
    else:
        epidermis_iou_oct, dice, best_mask = single_or_multiple_predictions(epidermis_mask, oct_mask, EPIDERMIS,
                                                                            dont_care_mask=dont_care_mask)
        print(f"OCT iou: {epidermis_iou_oct}.")
        print(f"OCT dice: {dice}.")
        total_iou_oct[EPIDERMIS] += epidermis_iou_oct
        total_dice_oct[EPIDERMIS] += dice
        df.loc[image_name, "iou_oct"] = epidermis_iou_oct
        df.loc[image_name, "dice_oct"] = dice
        df.loc[image_name, "nclicks_oct"] = n_points_used

        if visualize_pred_vs_gt_oct:
            visualize_prediction(best_mask, cropped_histology_gt, dont_care_mask, cropped_oct_image, dice, image_name, output_image_dir,
                                 prompts, ext="oct_pred")
    return prompts

def segment_oct_merge_mask_and_input(image_path, epidermis_mask, image_name, dont_care_mask):
    """ This one takes the prediction mask and oct input, and just merges them."""
    global output_image_dir, total_iou_vhist, total_dice_vhist
    fpath = f'{os.path.join(output_image_dir, image_name)}_predicted_mask_oct.npy'
    best_mask = numpy.load(fpath)
    crop_args_path = f'{os.path.join(output_image_dir, image_name)}_oct_crop_args.pickle'
    crop_args = pickle.load(open(crop_args_path, 'rb'))
    dont_care_mask = crop(dont_care_mask, **crop_args)
    path = f'{os.path.join(output_image_dir, image_name)}_cropped_oct_image.png'
    cropped_oct_image = cv2.imread(path)

    if visualize_pred_vs_gt_oct:
        visualize_prediction(best_mask, None, dont_care_mask, cropped_oct_image, None, image_name, output_image_dir,
                             None, ext="oct_pred")
    return None


def segment_vhist(image_path, epidermis_mask, image_name, dont_care_mask, prompts):
    global output_image_dir, total_iou_vhist, total_dice_vhist
    # v. histology segmentation
    print("virtual histology segmentation")
    path = f'{os.path.join(output_image_dir, image_name)}_cropped_vhist_image.png'
    cropped_vhist_mask, cropped_vhist, cropped_vhist_mask_true, cropped_oct_image, n_points_used, warped_vhist_mask_true, prompts, crop_args = predict_oct(
        image_path, epidermis_mask, args=args, weights_path=CHECKPOINT_PATH, create_vhist=create_virtual_histology,
        output_vhist_path=path, prompts = prompts)
    fpath = f'{os.path.join(output_image_dir, image_name)}_predicted_mask_vhist.npy'
    with open(fpath, 'wb+') as f:
        numpy.save(f,cropped_vhist_mask[0]) #a = numpy.load(fpath)
    # cropped_vhist_mask_true = crop(warped_vhist_mask_true, **crop_args)
    crop_args_path = f'{os.path.join(output_image_dir, image_name)}_vhist_crop_args.pickle'
    with open(crop_args_path, 'wb') as file:
        pickle.dump(crop_args, file)
    dont_care_mask = crop(dont_care_mask, **crop_args)
    if visualize_input_vhist:
        plt.figure(figsize=(5, 5))
        cropped_vhist = cv2.cvtColor(cropped_vhist, cv2.COLOR_BGR2RGB)
        plt.imshow(cropped_vhist)
        cropped_vhist = cv2.cvtColor(cropped_vhist, cv2.COLOR_BGR2RGB)
        show_mask(cropped_vhist_mask_true, plt.gca(), color_arr= COLORS.GT)
        plt.axis('off')
        plt.suptitle(f"Generated vhist and ground truth mask")
        plt.title(f"name {image_name}")
        plt.savefig(f'{os.path.join(output_image_dir, image_name)}_input_vhist.png')
        plt.close('all')

    if len(cropped_vhist_mask) == 0:
        print(f"Could not segment {image_path}.")
        return

    if not ANNOTATED_DATA:
        return
    epidermis_iou_vhist, dice, best_mask = single_or_multiple_predictions(cropped_vhist_mask_true, cropped_vhist_mask,
                                                                          EPIDERMIS, dont_care_mask=dont_care_mask)
    if best_mask is None:
        print(f"Could not calculate iou for {image_path}.")
        return

    # get bbox
    # divide mask_true by bbox area
    bbox = bounding_rectangle(epidermis_mask)
    bbox_area = bbox[2] * bbox[3]
    ntrue = numpy.unique(epidermis_mask, return_counts=True)[1][1]
    target_size_rel = ntrue / bbox_area

    print(f"v. histology iou: {epidermis_iou_vhist}.")
    print(f"v. histology dice: {dice}.")
    df.loc[image_name, "iou_vhist"] = epidermis_iou_vhist
    df.loc[image_name, "dice_vhist"] = dice
    df.loc[image_name, "target_size_rel"] = target_size_rel
    df.loc[image_name, "nclicks_vhist"] = n_points_used
    total_iou_vhist[EPIDERMIS] += epidermis_iou_vhist
    total_dice_vhist[EPIDERMIS] += dice

    if visualize_pred_over_vhist:
        visualize_prediction(best_mask, cropped_vhist_mask_true, dont_care_mask, cropped_vhist, dice, image_name, output_image_dir,
                             prompts, ext="vhist_pred")
        visualize_prediction(best_mask, cropped_vhist_mask_true, dont_care_mask,cropped_oct_image, dice, image_name, output_image_dir,
                             prompts, ext="vhist_pred_over_oct")

# def segment_vhist_merge_blah_blah(image_path, epidermis_mask, image_name, dont_care_mask, prompts):
#     """ This one takes the prediction mask and oct input, and just merges them."""
#     global output_image_dir, total_iou_vhist, total_dice_vhist
#     fpath = f'{os.path.join(output_image_dir, image_name)}_predicted_mask_vhist.npy'
#     # with open(fpath, 'wb+') as f:
#     #     numpy.save(f,cropped_vhist_mask[0]) #a = numpy.load(fpath)
#     best_mask = numpy.load(fpath)
#     vhist_crop_args_path = f'{os.path.join(output_image_dir, image_name)}_vhist_crop_args.pickle'
#     # with open(crop_args_path, 'wb') as file:
#     #     pickle.dump(crop_args, file)
#     crop_args = pickle.load(open(vhist_crop_args_path, 'rb'))
#     dont_care_mask = crop(dont_care_mask, **crop_args)
#     path = f'{os.path.join(output_image_dir, image_name)}_cropped_vhist_image.png'
#     cropped_vhist_image = cv2.imread(path)
#     oct_image = cv2.imread(image_path)
#     oct_crop_args_path = f'{os.path.join(output_image_dir, image_name)}_oct_crop_args.pickle'
#     crop_args = pickle.load(open(oct_crop_args_path, 'rb'))
#     cropped_oct_image = crop(oct_image, **crop_args)
#     if visualize_pred_over_vhist:
#         visualize_prediction(best_mask, None, dont_care_mask, cropped_vhist_image, None, image_name, output_image_dir,
#                              prompts, ext="vhist_pred")
#         visualize_prediction(best_mask, None, dont_care_mask,cropped_oct_image, None, image_name, output_image_dir,
#                              prompts, ext="vhist_pred_over_oct")


def does_column_exist(oct_fname, domain_dice_str): #domain_dice_str = "dice_oct" | "dice_vhist" | "dice_histology"
    sample_name = extract_filename_prefix(oct_fname)
    row = df.loc[sample_name]
    return domain_dice_str in row.index and not pandas.isna(row[domain_dice_str])

def main(args):
    global roboflow_next_img, df, output_image_dir, total_dice_oct, total_dice_vhist, total_iou_oct, total_iou_vhist, \
        total_iou_histology, total_dice_histology, total_samples_oct, total_samples_vhist, total_samples_histology

    download_images_and_masks()
    # Get the list of image files
    image_files = [f for f in os.listdir(roboflow_annot_dataset_dir) if f.endswith(".jpg")]
    image_files.sort()
    total_iou_vhist = {EPIDERMIS: 0}  # DERMIS:0 , # IOU for each class
    total_iou_oct = {EPIDERMIS: 0}
    total_iou_histology = {EPIDERMIS: 0}
    total_dice_vhist = {EPIDERMIS: 0}
    total_dice_oct = {EPIDERMIS: 0}
    total_dice_histology = {EPIDERMIS: 0}
    total_samples_vhist = 0
    total_samples_oct = 0
    total_samples_histology = 0
    path_to_annotations = os.path.join(roboflow_annot_dataset_dir, "_annotations.coco.json")
    from pylabel import importer
    dataset = importer.ImportCoco(path_to_annotations, path_to_images=roboflow_annot_dataset_dir, name="zero_shot_oct")
    csv_path = os.path.join(args.output_dir, "iou_scores.csv")
    csv_exists = os.path.exists(csv_path)
    if csv_exists:
        shutil.copyfile(csv_path, csv_path + ".previous")

    if skip_existing_images and csv_exists:
        df = pd.read_csv(csv_path, index_col='Unnamed: 0')
    else:
        index_array = [extract_filename_prefix(file) for file in image_files]
        df = pd.DataFrame({"iou_vhist": numpy.nan, "iou_oct": numpy.nan,"iou_histology": numpy.nan,
                           "dice_oct": numpy.nan, "dice_vhist": numpy.nan,
                           "dice_histology": numpy.nan, }, index=index_array)

    take_first_n_images =  -1
    output_image_dir = args.output_dir
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    i = 0

    if take_first_n_images > 0:
        image_files = image_files[:take_first_n_images]
    for oct_fname in tqdm(image_files):
        if single_image_to_segment is not None and not extract_filename_prefix(oct_fname).startswith(
                single_image_to_segment):
            continue
        if patient_to_skip is not None:
            skip_sample = False
            for patient in patient_to_skip:
                if extract_filename_prefix(oct_fname).startswith(patient):
                    skip_sample = True
                    break
            if skip_sample:
                continue

        i += 1
        prompts = None
        image_name = extract_filename_prefix(oct_fname)
        print(f"\nimage number {i}: {image_name}")
        image_path = os.path.join(roboflow_annot_dataset_dir, oct_fname)
        roboflow_next_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if ANNOTATED_DATA:
            oct_data = dataset.df[dataset.df.img_filename == oct_fname]
            epidermis_data = oct_data[oct_data.cat_name == "epidermis"].ann_segmentation.values[0][0]
            epidermis_mask = coco_mask_to_numpy(roboflow_next_img.shape[:2], epidermis_data)
            if 'hair' in oct_data.cat_name.unique():
                dont_care_data = oct_data[oct_data.cat_name == "hair"].ann_segmentation.values[0][0]
                dont_care_mask = coco_mask_to_numpy(roboflow_next_img.shape[:2], dont_care_data)
                epidermis_mask = epidermis_mask & (~dont_care_mask)
            else:
                dont_care_mask = None
        else:
            epidermis_mask = None
            dont_care_mask = None


        if visualize_input_gt:
            plt.figure(figsize=(5, 5))
            plt.imshow(roboflow_next_img)
            show_mask(epidermis_mask, plt.gca(), color_arr= COLORS.GT)
            plt.axis('off')
            plt.suptitle(f"Input oct and ground truth mask")
            plt.title(f"{image_name}")
            plt.savefig(f'{os.path.join(output_image_dir, image_name)}_input_gt.png')
            plt.close('all')
        skip_oct = skip_existing_images and does_column_exist(oct_fname, "dice_oct")
        if not skip_oct:
            prompts = segment_oct(image_path, epidermis_mask, image_name, dont_care_mask)
            total_samples_oct += 1
        else:
            print(f"skipping oct segmentation")
        if segment_real_hist:
            skip_hist = skip_existing_images and does_column_exist(oct_fname, "dice_histology")
            if not skip_hist:
                file_name = image_name[:-1] + "B.jpg"
                image_path_hist = os.path.join(raw_oct_dataset_dir, file_name)
                # histology segmentation
                segment_histology(image_path_hist, epidermis_mask, image_name, dont_care_mask, prompts)
                total_samples_histology += 1
            else:
                print(f"skipping histology segmentation")
        if create_virtual_histology:
            skip_vhist = skip_existing_images and does_column_exist(oct_fname, "dice_vhist")
            if not skip_vhist:
                segment_vhist(image_path, epidermis_mask, image_name, dont_care_mask, prompts)
                total_samples_vhist += 1
            else:
                print(f"skipping virtual histology segmentation")
        df.to_csv(os.path.join(output_image_dir, 'iou_scores.csv'), index=True)
    handle_stats(df, output_image_dir, total_dice_oct, total_dice_vhist, total_dice_histology, total_iou_oct, total_iou_vhist,
                 total_samples_oct, total_samples_vhist, total_samples_histology)


def handle_stats(df, output_image_dir, total_dice_oct, total_dice_vhist, total_dice_histology, total_iou_oct, total_iou_vhist,
                 total_samples_oct, total_samples_vhist, total_samples_histology):
    average_iou = total_iou_vhist[
                      EPIDERMIS] / total_samples_vhist  # sum all ious divided by (number of images * number of classes).
    print(f"Average IoU with virtual histology: {average_iou}")
    average_iou_oct = total_iou_oct[EPIDERMIS] / total_samples_oct
    print(f"Average IoU without virtual histology: {average_iou_oct}")
    average_iou_histolgy = total_iou_histology[EPIDERMIS] / total_samples_histology
    print(f"Average IoU with real histology: {average_iou_histolgy}")
    average_dice = total_dice_vhist[
                       EPIDERMIS] / total_samples_vhist
    print(f"Average dice with virtual histology: {average_dice}")
    average_dice_oct = total_dice_oct[EPIDERMIS] / total_samples_oct
    print(f"Average dice without virtual histology: {average_dice_oct}")
    average_dice_histology = total_dice_histology[EPIDERMIS] / total_samples_histology
    print(f"Average dice with real histology: {average_dice_histology}")
    from scipy.stats import ttest_ind
    array1 = df["dice_oct"].values
    array2 = df["dice_vhist"].values
    # Perform a two-sample t-test
    t_statistic, p_value = ttest_ind(array1, array2, nan_policy="omit", equal_var=False, )
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

def visualize_prediction_over_oct(mask, cropped_oct_image):
    plt.figure(figsize=(5, 5))
    # cropped_oct_image = cv2.cvtColor(cropped_oct_image, cv2.COLOR_BGR2RGB)
    plt.imshow(cropped_oct_image)
    # cropped_oct_image = cv2.cvtColor(cropped_oct_image, cv2.COLOR_BGR2RGB)
    c1 = show_mask(mask, plt.gca(), color_arr= COLORS.GT)
    # c2 = show_mask(cropped_histology_gt, plt.gca(), random_color=True, alpha=0.2)
    plt.axis('off')
    # plt.suptitle(f"oct segmentation w/o vhist: iou {epidermis_iou_oct:.2f}")
    plt.savefig(f'image_name.png', bbox_inches='tight', pad_inches=0)
    # save_diff_image(best_mask, cropped_histology_gt, fpath)
    plt.close('all')
    return f'image_name.png'

def visualize_prediction(best_mask, epidermis_mask, dont_care_mask, cropped_oct_image, dice, image_name, output_image_dir,
                         prompts, ext):
    best_mask = best_mask.astype(bool)
    overlay = cropped_oct_image.copy()
    overlay[best_mask] = (255,128,0)
    alpha = 0.6
    overlayed_image = cv2.addWeighted(overlay, alpha, cropped_oct_image, 1 - alpha, 0)
    fpath = f'{os.path.join(output_image_dir, image_name)}_{ext}.png'
    cv2.imwrite(fpath, overlayed_image)

def visualize_prediction_prev(best_mask, epidermis_mask, dont_care_mask, cropped_oct_image, dice, image_name, output_image_dir,
                         prompts, ext):
    plt.figure(figsize=(5, 5))
    cropped_oct_image = cv2.cvtColor(cropped_oct_image, cv2.COLOR_BGR2RGB)
    plt.imshow(cropped_oct_image)
    cropped_oct_image = cv2.cvtColor(cropped_oct_image, cv2.COLOR_BGR2RGB)
    c1 = show_mask(best_mask, plt.gca(), color_arr= COLORS.GT)
    # c2 = show_mask(epidermis_mask, plt.gca(), color_arr= COLORS.EPIDERMIS, outline=True)
    if dont_care_mask is not None:
        c3 = show_mask(dont_care_mask, plt.gca(), color_arr=COLORS.DONT_CARE)
    # c2 = show_mask(cropped_histology_gt, plt.gca(), random_color=True, alpha=0.2)
    plt.axis('off')
    # plt.suptitle(f"oct segmentation w/o vhist: iou {epidermis_iou_oct:.2f}")


    if dice is not None:
        text_to_display = f"dice {dice:.2f}"
        plt.text(0.02, 0.9, text_to_display, color='white', fontsize=12, transform=plt.gca().transAxes)
    if args.point:
        add_pts, remove_pts = prompts["add"], prompts["remove"]
        # overlay points
        plt.scatter(remove_pts[:, 0], remove_pts[:, 1], color='red', marker='o', s=10)
        plt.scatter(add_pts[:, 0], add_pts[:, 1], color='lightgreen', marker='+', s=15)
    if args.box and prompts is not None:
        # overlay box
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
    # save_diff_image(best_mask, cropped_histology_gt, fpath)
    plt.close('all')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process point, box, and grid arguments.")
    parser.add_argument("--output_dir", help="Specify output directory, e.g. './images/point_prediction' ")
    parser.add_argument("--take_first_n", help="take first n images", default=-1, type=int)
    parser.add_argument("--npoints", help="number_of_prediction_points", default=10, type=int)
    parser.add_argument("--remove_output_dir", action="store_true", help="remove output dir before running the script")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--point", action="store_true", help="Specify a point.")
    group.add_argument("--box", action="store_true", help="Specify a box.")
    group.add_argument("--grid", action="store_true", help="Specify a grid.")
    args = parser.parse_args()
    print("run test_iou...prompts.py instead")
    raise Exception("unifying both run files")
    if MEDSAM and args.point:
        raise Exception("MedSam does not support points")
    if not args.point and not args.box and not args.grid:
        print("Please specify one of --point, --box, or --grid.")
    elif not args.output_dir:
        print("Please specify output dir.")
    else:
        if args.remove_output_dir and os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        main(args)
