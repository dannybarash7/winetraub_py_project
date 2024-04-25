import os
import sys

import cv2
import matplotlib as mpl
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from skimage import transform

from OCT2Hist_UseModel.utils.masking import apply_closing_operation
from zero_shot_segmentation.consts import MEDSAM, SAMMED_2D, SAM, INTERACTIVE_POINT_PREDICTION
from zero_shot_segmentation.zero_shot_utils.utils import bounding_rectangle, get_center_of_mass, \
    expand_bounding_rectangle

sys.path.append("./OCT2Hist_UseModel/")

if SAM or MEDSAM:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
if SAMMED_2D:
    from SAM_Med2D.segment_anything import sam_model_registry as sammed_model_registry
    from SAM_Med2D.segment_anything.predictor_sammed import SammedPredictor
segmenter = None


def run_gui(img, weights_path, args, gt_mask=None, auto_segmentation=True, prompts=None, dont_care_mask = None):
    global segmenter
    if img is None:
        raise Exception("Image file not found.")

    # if img.ndim == 3:
    #     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     a=1

    elif img.ndim == 2:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    if args.point:
        segmenter = Segmenter(img, weights_path, auto_segmentation=auto_segmentation, gt_mask=gt_mask,
                              point_prediction_flag=True, npoints=args.npoints, prompts=prompts, dont_care_mask = dont_care_mask)
    elif args.box:
        segmenter = Segmenter(img, weights_path, auto_segmentation=auto_segmentation, gt_mask=gt_mask,
                              box_prediction_flag=True)
    elif args.grid:
        segmenter = Segmenter(img, weights_path, auto_segmentation=auto_segmentation, gt_mask=gt_mask,
                              grid_prediction_flag=True)
    if not auto_segmentation:
        plt.show(block=True)
    return segmenter


def get_point_grid():
    # Define the number of points on each axis
    num_x_points = 32
    num_y_points = 32
    width = 1024
    height = 512
    start_y = int(0.1 * height)
    end_y = int(0.6 * height)
    # Create 1D arrays for x and y coordinates
    x = np.linspace(0, width, num_x_points)
    y = np.linspace(start_y, end_y, num_y_points)
    x = x[1:-1]
    y = y[1:-1]
    # Use numpy.meshgrid to create a grid of (x, y) points
    xx, yy = np.meshgrid(x, y)
    xx = xx / width
    yy = yy / height
    # Stack the (x, y) points into a single array of (x, y) pairs
    points = np.stack((xx, yy), axis=-1)
    points = points.reshape(-1, 2)
    # Convert the numpy arrays to a list of np.ndarray objects
    points_list = [points]
    """
    [array([[0.015625, 0.015625],
        [0.046875, 0.015625],
        [0.078125, 0.015625],
        ...,
        [0.921875, 0.984375],
        [0.953125, 0.984375],
        [0.984375, 0.984375]])]
    """
    return points_list


class Segmenter():
    _predictor = None

    def __init__(self, img, weights_path, auto_segmentation, npoints=0, box_prediction_flag=False,
                 point_prediction_flag=False, grid_prediction_flag=False, gt_mask=None, remaining_points=20,
                 prompts=None, dont_care_mask = None):
        """

        :param img:
        :param weights_path:
        :param npoints: number of points to sample from mask fg, same number will be used for fg.
        :param box_prediction_flag:
        :param point_prediction_flag:
        :param auto_segmentation: if automatic is on, and grid_prediction_flag is on, it's full grid, multimask prediction. if box: it's a tight box around the gt. If point: random bg point from the gt box.
        :param gt_mask:
        """
        if MEDSAM:
            self.device = "cpu"
        else:
            self.device = "mps"
        self.img = img
        self.min_mask_region_area = 500
        self.npoints = npoints
        self.remaining_points = remaining_points
        self.init_points = npoints
        self.box_prediction_flag = box_prediction_flag
        self.point_prediction_flag = point_prediction_flag
        self.grid_prediction_flag = grid_prediction_flag
        self.auto_segmentation = auto_segmentation
        self.gt_mask = gt_mask
        self.dont_care_mask = dont_care_mask
        self.init_points = npoints
        self.prompts = prompts

        if Segmenter._predictor is None:  # init predictor
            if SAMMED_2D:
                from argparse import Namespace
                args = Namespace()
                args.image_size = 256
                args.encoder_adapter = True
                args.sam_checkpoint = "/Users/dannybarash/Code/oct/medsam/sam-med2d/OCT2Hist_UseModel/SAM_Med2D/pretrain_model/sam-med2d_b.pth"
                model = sammed_model_registry["vit_b"](args)
                self.sam = model  # sam_model_registry["vit_h"](checkpoint=weights_path)
            if MEDSAM:
                self.sam = sam_model_registry["vit_b"](checkpoint=weights_path)
            if SAM:
                self.sam = sam_model_registry["vit_h"](checkpoint=weights_path)
            if not grid_prediction_flag:
                if MEDSAM or SAM:
                    self.predictor = SamPredictor(self.sam)
                if SAMMED_2D:
                    self.predictor = SammedPredictor(model)
                self.predictor.set_image(self.img)
            else:
                self.predictor = SamAutomaticMaskGenerator(
                    self.sam,
                    pred_iou_thresh=0.0,  # relevant
                    stability_score_thresh=0.0,
                    box_nms_thresh=1.0,  # relevant
                    crop_n_layers=0,  # relevant
                    crop_nms_thresh=1.0,  # relevant
                )
                # point_grids = get_point_grid()
                # self.predictor = SamAutomaticMaskGenerator(
                #     self.sam,
                #     points_per_side=32,  # 32 relevant
                #     points_per_batch=64,
                #     pred_iou_thresh=0.0,  # relevant
                #     stability_score_thresh=0.0,  # relevant (default is 0.95)
                #     stability_score_offset=1.0,  # relevant
                #     box_nms_thresh=1.0,  # relevant
                #     crop_n_layers=0,  # relevant
                #     crop_nms_thresh=1.0,  # relevant
                #     crop_overlap_ratio=512 / 1500,  # relevant
                #     crop_n_points_downscale_factor=1,  # relevant
                #     point_grids=None,  # point_grids,
                #     min_mask_region_area=0,  # relevant, default is 0
                #     # output_mode: str = "binary_mask",
                # )
            # save for later
            Segmenter._predictor = self.predictor
        else:  # predictor saved from previous calls
            self.predictor = Segmenter._predictor

        if not grid_prediction_flag:
            print("Creating image embeddings ... ", end="")
            self.predictor.set_image(self.img)
            print("Done")

        self.color_set = set()
        self.current_color = self.pick_color()
        self.add_xs, self.add_ys, self.rem_xs, self.rem_ys, self.trace = [], [], [], [], []

        self.fig, self.ax = plt.subplots(figsize=(10 * (self.img.shape[1] / max(self.img.shape)),
                                                  10 * (self.img.shape[0] / max(self.img.shape))))
        self.fig.suptitle(f'Segment Anything GUI: {self.remaining_points} points remain', fontsize=16)
        self.ax.set_title("Press 'h' to show/hide commands.", fontsize=10)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.im = self.ax.imshow(self.img, cmap=mpl.cm.gray)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.ax.autoscale(False)
        self.label = 1
        self.add_plot, = self.ax.plot([], [], 'o', markerfacecolor='green', markeredgecolor='black', markersize=5)
        self.rem_plot, = self.ax.plot([], [], 'x', markerfacecolor='red', markeredgecolor='red', markersize=5)
        self.mask_data = np.zeros((self.img.shape[0], self.img.shape[1], 4), dtype=np.uint8)
        self.masks = []

        for i in range(3):
            self.mask_data[:, :, i] = self.current_color[i]
        self.mask_plot = self.ax.imshow(self.mask_data)
        self.prev_mask_data = np.zeros((self.img.shape[0], self.img.shape[1], 4), dtype=np.uint8)
        self.prev_mask_plot = self.ax.imshow(self.prev_mask_data)
        self.contour_plot, = self.ax.plot([], [], color='black')
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self.show_help_text = False
        self.help_text = plt.text(0.5, 0.5, '', horizontalalignment='center', verticalalignment='center',
                                  transform=self.ax.transAxes)
        self.opacity = 120  # out of 255
        self.global_masks = np.zeros((self.img.shape[0], self.img.shape[1]), dtype=np.uint8)
        self.last_mask = np.zeros((self.img.shape[0], self.img.shape[1]), dtype=bool)  # to undo
        self.full_legend = []
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        if self.auto_segmentation:
            self.get_mask()

    def save_annotation(self, labels_file_outpath):
        dir_path = os.path.split(labels_file_outpath)[0]
        if dir_path != '' and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        cv2.imwrite(labels_file_outpath, self.global_masks)
        print(f"Saved annotation at {labels_file_outpath}")

    def pick_color(self):
        while True:
            color = tuple(np.random.randint(low=0, high=255, size=3).tolist())
            if color not in self.color_set:
                self.color_set.add(color)
                return color

    def _on_key(self, event):
        if event.key == 'z':
            if self.remaining_points < self.init_points:
                self.remaining_points += 1
                self.fig.suptitle(f'Segment Anything GUI: {self.remaining_points} points remain', fontsize=16)
            self.undo()


        elif event.key == 'enter':
            self.new_tow()

        elif event.key == 'escape':  # save for notebooks
            plt.close(self.fig)

        elif event.key == 'h':
            if not self.show_help_text:
                self.help_text.set_text('• \'left click\': select a point inside object to label\n'
                                        '• \'right click\': select a point to exclude from label\n'
                                        '• \'enter\': confirm current label and create new\n'
                                        '• \'z\': undo point\n'
                                        '• \'esc\': close and save')
                self.help_text.set_bbox(dict(facecolor='white', alpha=1, edgecolor='black'))
                self.show_help_text = True
            else:
                self.help_text.set_text('')
                self.show_help_text = False
            self.fig.canvas.draw()

    def _on_click(self, event):
        if self.remaining_points > 0:
            self.remaining_points -= 1
            self.fig.suptitle(f'Segment Anything GUI: {self.remaining_points} points remain', fontsize=16)
        else:
            return
        if event.inaxes != self.ax and (event.button in [1, 3]): return
        x = int(np.round(event.xdata))
        y = int(np.round(event.ydata))

        if event.button == 1:  # left click
            self.trace.append(True)
            self.add_xs.append(x)
            self.add_ys.append(y)
            self.show_points(self.add_plot, self.add_xs, self.add_ys)

        else:  # right click
            self.trace.append(False)
            self.rem_xs.append(x)
            self.rem_ys.append(y)
            self.show_points(self.rem_plot, self.rem_xs, self.rem_ys)

        self.get_mask()

    def show_points(self, plot, xs, ys):
        plot.set_data(xs, ys)
        self.fig.canvas.draw()

    def clear_mask(self):
        self.contour_plot.set_data([], [])
        self.mask_data.fill(0)
        self.mask_plot.set_data(self.mask_data)
        self.fig.canvas.draw()

    def get_mask_for_manual_rect(self):
        if len(self.add_xs) >= 2 and len(self.add_xs) % 2 == 0:
            last_2_xs = self.add_xs[-2:]
            last_2_xs.sort()
            last_2_ys = self.add_ys[-2:]
            last_2_ys.sort()
            user_box = np.array(list([last_2_xs[0], last_2_ys[0], last_2_xs[1], last_2_ys[1]]))
            masks, _, _ = self.predictor.predict(box=user_box, multimask_output=False)
            return masks

    def transform_img(self, img_np, box_np):
        medsam_model = self.predictor.model
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np
        H, W, _ = img_3c.shape
        # %% image preprocessing
        img_1024 = transform.resize(
            img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)
        # convert the shape to (3, H, W)
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_1024_tensor)
        box_np = np.array([box_np])
        box_1024 = box_np / np.array([W, H, W, H]) * 1024
        return image_embedding, box_1024

    @torch.no_grad()
    def medsam_inference(self, img, box):
        H, W, _ = img.shape
        img_embed, box_1024 = self.transform_img(img, box)
        medsam_model = self.predictor.model
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]  # (B, 1, 4)
        sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, _ = medsam_model.mask_decoder(
            image_embeddings=img_embed,  # (B, 256, 64, 64)
            image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

        low_res_pred = F.interpolate(
            low_res_pred,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # (1, 1, gt.shape)
        low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
        return medsam_seg

    def get_mask_for_auto_rect(self):
        self.user_box = bounding_rectangle(self.gt_mask)
        if self.gt_mask is None:
            print("Missing input to auto rect box prediction")
            return None
        if MEDSAM:
            masks = self.medsam_inference(self.img, self.user_box)
            masks = np.expand_dims(masks, 0)
        #     user_box_to_sam_med = self.user_box / np.array([1024,  512, 1024,  512]) * 1024
        #     masks, _, _ = self.predictor.predict(box=self.user_box, multimask_output=False)
        else:
            masks, qualities, low_res_mask_inputs = self.predictor.predict(box=self.user_box, multimask_output=False)
            # i = qualities.argmax()
            # masks = masks[i:i+1,:,:]

        return masks

    def points_in_rectangle(self, points, user_box):
        """
        Get points from the 2D array that are contained within the specified rectangle.

        Parameters:
        - points: 2D array of points where each row is (x, y).
        - x1, y1, x2, y2: Bounding rectangle coordinates.

        Returns:
        - Array of points within the rectangle.
        """
        x1, y1, x2, y2 = user_box
        mask = (points[:, 1] >= x1) & (points[:, 1] <= x2) & (points[:, 0] >= y1) & (points[:, 0] <= y2)
        return points[mask]

    def find_max_distance_pixel(self,mask):
        # Convert the boolean mask to an 8-bit type
        mask = mask.astype(np.uint8) * 255

        # Calculate distance transform
        distance_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

        # Find the coordinates of the pixel with the maximum distance value
        max_distance_index = np.unravel_index(np.argmax(distance_transform), distance_transform.shape)

        return max_distance_index

    def get_mask_for_auto_point(self):
        if self.prompts is None:
            gt_bbox = bounding_rectangle(self.gt_mask)
            # Note: all points returning from argwhere are in [y,x] (row,column) format.
            twoX_gt_bbox = expand_bounding_rectangle(gt_bbox, self.img.shape[:2])
            # all background points
            neg_points = np.argwhere(~(self.gt_mask | self.dont_care_mask))
            # filter negative (background) points in the bounding box of 2x box
            neg_points_in_2x_gt_bbox = self.points_in_rectangle(neg_points, twoX_gt_bbox)
            remove_pts = self.sample_points(neg_points_in_2x_gt_bbox)
            pos_points = np.argwhere(self.gt_mask)
            # pos_points_in_gt_bbox = self.points_in_rectangle(pos_points, user_box)
            # assert len(pos_points_in_gt_bbox) + len(neg_points_in_gt_bbox) == (user_box[2]-user_box[0] ) * (user_box[3]-user_box[1])
            add_pts = self.sample_points(pos_points)
            com = get_center_of_mass(self.gt_mask)
            max_distance_pixel = self.find_max_distance_pixel(self.gt_mask)
            if self.gt_mask[com[0], com[1]]:  # if center of mass is in forground, overwrite the first point with it
                add_pts[0] = [com[1], com[0]]  # com is from mat indices, [row,col], while add_pts is [x,y] format.


        else:
            add_pts = self.prompts["add"]
            remove_pts = self.prompts["remove"]

        self.add_pts = add_pts
        self.remove_pts = remove_pts
        mask_input = None

        if INTERACTIVE_POINT_PREDICTION:  # iterative
            all_points = np.concatenate([self.add_pts, self.remove_pts])
            all_labels = np.array([1] * len(self.add_pts) + [0] * len(self.remove_pts))
            points_so_far = []
            labels_so_far = []
            for point, label in zip(all_points, all_labels):
                points_so_far.append(point)
                labels_so_far.append(label)
                masks, scores, logits = self.predictor.predict(point_coords=np.array(points_so_far),
                                                               point_labels=np.array(labels_so_far),
                                                               multimask_output=True, mask_input=mask_input)
                if SAMMED_2D:
                    mask_input = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float))
                else:
                    mask_input = logits[np.argmax(scores), :, :]
                    mask_input = mask_input[None, :, :]
        else:
            masks, scores, logits = self.predictor.predict(point_coords=np.concatenate([self.add_pts, self.remove_pts]),
                                                           point_labels=np.array(
                                                               [1] * len(self.add_pts) + [0] * len(self.remove_pts)),
                                                           multimask_output=False, mask_input=mask_input)
            return masks

        return masks

    def sample_points(self, points_in_mask):
        num_rows = points_in_mask.shape[0]
        random_index = np.random.randint(0, num_rows, size=self.npoints)
        points_yx_in_mask = points_in_mask[random_index]
        points_xy_in_mask = points_yx_in_mask[:, ::-1]
        return points_xy_in_mask

    def handle_single_mask(self, masks):
        mask = masks[0, :, :].astype(np.uint8)
        mask[self.global_masks > 0] = 0
        mask = self.remove_small_regions(mask, self.min_mask_region_area, "holes")
        mask = self.remove_small_regions(mask, self.min_mask_region_area, "islands")
        mask = apply_closing_operation(mask)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        xs, ys = [], []
        for contour in contours:  # nan to disconnect contours
            xs.extend(contour[:, 0, 0].tolist() + [np.nan])
            ys.extend(contour[:, 0, 1].tolist() + [np.nan])
        self.contour_plot.set_data(xs, ys)
        self.masks.append(mask)
        self.mask_data[:, :, 3] = mask * self.opacity
        self.mask_plot.set_data(self.mask_data)
        self.fig.canvas.draw()

    def handle_multimask(self, masks):
        # used in grid based prediction.
        # TODO instead of a for loop, calculate intersection ious and drop the non maximal one.
        # like in calculate_iou_for_multiple_predictions(mask_true, mask_predictions, class_id)
        for i, mask in enumerate(masks):
            mask = mask["segmentation"]
            mask = mask.astype(np.uint8)
            mask[self.global_masks > 0] = 0
            mask = self.remove_small_regions(mask, self.min_mask_region_area, "holes")
            mask = self.remove_small_regions(mask, self.min_mask_region_area, "islands")
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            xs, ys = [], []
            for contour in contours:  # nan to disconnect contours
                xs.extend(contour[:, 0, 0].tolist() + [np.nan])
                ys.extend(contour[:, 0, 1].tolist() + [np.nan])
            self.contour_plot.set_data(xs, ys)
            self.masks.append(mask)
            self.mask_data[:, :, 3] = mask * self.opacity

    def get_mask(self):
        if self.box_prediction_flag and not self.auto_segmentation:
            mask = self.get_mask_for_manual_rect()
            if mask is not None:
                self.handle_single_mask(mask)
        if self.point_prediction_flag and not self.auto_segmentation:
            raise Exception("manual point_prediction_flag WIP")
        elif self.box_prediction_flag and self.auto_segmentation:
            mask = self.get_mask_for_auto_rect()
            self.handle_single_mask(mask)
        elif self.point_prediction_flag and self.auto_segmentation:
            mask = self.get_mask_for_auto_point()
            self.handle_single_mask(mask)
        elif self.grid_prediction_flag and self.auto_segmentation:
            masks = self.predictor.generate(self.img)
            self.handle_multimask(masks)
        else:
            raise Exception("Bad flag combination")

    def undo(self):
        if len(self.trace) == 0:  # undo last mask
            self.global_masks[self.last_mask] = 0
            self.prev_mask_data[:, :, 3][self.last_mask] = 0
            self.prev_mask_plot.set_data(self.prev_mask_data)
            self.label -= 1
            self.full_legend.pop()
            self.ax.legend(handles=self.full_legend, loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
            self.clear_mask()

        else:  # undo last point
            if self.trace[-1]:
                self.add_xs = self.add_xs[:-1]
                self.add_ys = self.add_ys[:-1]
                self.show_points(self.add_plot, self.add_xs, self.add_ys)
            else:
                self.rem_xs = self.rem_xs[:-1]
                self.rem_ys = self.rem_ys[:-1]
                self.show_points(self.rem_plot, self.rem_xs, self.rem_ys)

            self.trace.pop()

            if len(self.trace) != 0:
                self.get_mask()
            else:
                self.clear_mask()

    def new_tow(self):
        # clear points
        self.add_xs, self.add_ys, self.rem_xs, self.rem_ys, self.trace = [], [], [], [], []
        self.show_points(self.add_plot, self.add_xs, self.add_ys)
        self.show_points(self.rem_plot, self.rem_xs, self.rem_ys)

        mask = self.mask_data[:, :, 3] > 0
        self.global_masks[mask] = self.label
        self.last_mask = mask.copy()

        self.prev_mask_data[:, :, :3][mask] = self.current_color
        self.prev_mask_data[:, :, 3][mask] = 255
        self.prev_mask_plot.set_data(self.prev_mask_data)

        self.full_legend.append(Circle(1, color=np.array(self.current_color) / 255, label=f'{self.label}'))
        self.ax.legend(handles=self.full_legend, loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

        self.current_color = self.pick_color()
        self.label += 1

        for i in range(3):
            self.mask_data[:, :, i] = self.current_color[i]
        self.clear_mask()

    @staticmethod
    def remove_small_regions(mask, area_thresh, mode):
        """Function from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/utils/amg.py"""
        assert mode in ["holes", "islands"]
        correct_holes = mode == "holes"
        working_mask = (correct_holes ^ mask).astype(np.uint8)
        n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
        sizes = stats[:, -1][1:]  # Row 0 is background label
        small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
        if len(small_regions) == 0:
            return mask
        fill_labels = [0] + small_regions
        if not correct_holes:
            fill_labels = [i for i in range(n_labels) if i not in fill_labels]
            # If every region is below threshold, keep largest
            if len(fill_labels) == 0:
                fill_labels = [int(np.argmax(sizes)) + 1]
        mask = np.isin(regions, fill_labels)
        return mask


def run_gui_segmentation(img, weights_path, gt_mask, args, prompts, dont_care_mask):
    segmenter = run_gui(img, weights_path, args, gt_mask, prompts=prompts, dont_care_mask = dont_care_mask)
    segmenter.global_masks[segmenter.global_masks > 0] = 1
    points_used = segmenter.init_points - segmenter.remaining_points
    # Danny: masks is for all masks, global masks is for the unified fixes
    # segmenter.masks holds the predicted mask
    # in case of point based predictions:
    """
            self.add_pts = add_pts
            self.remove_pts = remove_pts
    """
    # hold the positive and negative points.
    # in case of rectangle based predictions, self.user_box contains it.

    if args.point:
        prompts = {"add": segmenter.add_pts, "remove": segmenter.remove_pts}
    else:
        prompts = {"box": segmenter.user_box}
    return segmenter.masks, points_used, prompts
