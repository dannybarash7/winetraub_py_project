# Segment skin from gel in H&E images utilizing facebook's SAM
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import segment_skin_in_oct_images

class SegmentSkinInHEImages:
  def __init__(self, sam):
      self.predictor = SamPredictor(sam)
      segment_skin_in_oct_images.setup_network()

  # This function segments skin from gel.
  # Inputs:
  #  he_image - H&E image loaded in RGB (N by M by 3)
  #  oct_image - if exists, utilize OCT for segmentation as it is more accurate
  #  visualize_results - set to True to create a visualization of the results
  # Outputs:
  #  binary_mask - True - for every region that is skin, false otherwise
  def segment_skin_from_gel(self, he_image, oct_image=None, visualize_results=False):
    if oct_image is None:
      # Run segmentation based on h&e
      return self._segment_skin_from_gel_he(he_image, visualize_results)
    else:
      # Segment using OCT
      mask = segment_skin_in_oct_images.run_network(oct_image)
      if visualize_results:
        self._visualize_results(he_image, mask, [], [])
      return mask

  # This function segments skin from gel using h&e image (without using OCT)
  # Inputs:
  #  he_image - H&E image loaded in RGB (N by M by 3)
  #  visualize_results - set to True to create a visualization of the results
  # Outputs:
  #  binary_mask - True - for every region that is skin, false otherwise
  def _segment_skin_from_gel_he(self, he_image, visualize_results=False):
    # Downscale image since tissue is something big and thus lower resultion is better
    scaled_image = cv2.resize(he_image, (
      int(he_image.shape[0]/4), int(he_image.shape[1]/4)))
    
    # Find points of interest
    points_array, points_label = self._compute_points_of_interest(scaled_image)
    
    # Do the SAM thing
    self.predictor.set_image(scaled_image)
    masks, scores, logits = self.predictor.predict(
      point_coords=points_array,
      point_labels=points_label,
      multimask_output=False,
      )
    mask = masks[0]

    # Scale mask & points up
    mask = cv2.resize(mask.astype(np.float32), (he_image.shape[0], he_image.shape[1]))
    mask = mask>0.5
    points_array = points_array*4

    # Make sure no black pixels are part of the mask
    im_gray = cv2.cvtColor(he_image, cv2.COLOR_RGB2GRAY)
    mask = mask*np.array(im_gray > 0)

    if visualize_results:
      self._visualize_results(he_image, mask, points_array, points_label)
    return mask

  # This function predicts points that are to be used for SAM
  # Inputs:
  #  he_image - H&E image in RGB
  # Outputs:
  #  points_array - an array of points (in pixels) [[x,y],[x,y],...]
  #  points_label - an array of the same size as points_array where value can be 1 for including the point, 0 for excluding
  def _compute_points_of_interest(self, he_image):
    # Create a mask that outlines the borders of th image, exclude darked out areas
    im_gray = cv2.cvtColor(he_image, cv2.COLOR_RGB2GRAY)
    mask = np.array(im_gray > 0)

    # Find center of mass of the main blob
    moments = cv2.moments(mask.astype(np.uint8))
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])

    # From center of mass, go down and right, find the x and y of the points that are still in the tissue (i.e. not in dark area)
    def go_down(pt,mask): # pt is [cx,cy]
      cx = int(pt[0])
      cy = int(pt[1])
      lowest_y = np.where(mask[:, cx])[0]
      lowest_y = lowest_y[len(lowest_y)-1]  
      return [cx, lowest_y-2]
    def go_up(pt,mask): # pt is [cx,cy]
      cx = int(pt[0])
      cy = int(pt[1])
      highest_y = np.where(mask[:, cx])[0]
      highest_y = highest_y[0]  
      return [cx, highest_y+2]
    def go_right(pt,mask): # pt is [cx,cy]
      cx = int(pt[0])
      cy = int(pt[1])
      rightest_x = np.where(mask[cy, :])[0]
      rightest_x = rightest_x[len(rightest_x)-1]
      return [rightest_x-2, cy]
    def go_left(pt,mask): # pt is [cx,cy]
      cx = int(pt[0])
      cy = int(pt[1])
      left_x = np.where(mask[cy, :])[0]
      left_x = left_x[0]
      return [left_x+2, cy]

    # Finish up by creating the points to be used
    points_array = np.array([
      [cx, cy], 
      go_down([cx, cy], mask),
      go_down(go_right([cx, cy], mask), mask),
      go_down(go_left([cx, cy], mask), mask),
      go_up([cx, cy], mask),
      go_up([cx+mask.shape[1]/4, cy], mask),
      go_up([cx-mask.shape[1]/4, cy], mask),
      go_up(go_right([cx, cy], mask), mask),
      go_up(go_left([cx, cy], mask), mask),
      ])
    points_label = np.array([1,1,1,1,0,0,0,0,0])

    return(points_array, points_label)

  # This function visualizes results
  def _visualize_results(self, he_image, mask, points_array, points_label):
    def show_points(coords, labels, ax, marker_size=375):
      if len(points_array) == 0:
        return
      pos_points = coords[labels==1]
      neg_points = coords[labels==0]
      ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
      ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    def show_mask(mask, ax, random_color=False):
      if random_color:
          color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
      else:
          color = np.array([30/255, 144/255, 255/255, 0.6])
      h, w = mask.shape[-2:]
      mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
      ax.imshow(mask_image)

    plt.figure(figsize=(5,5))
    plt.imshow(he_image)
    show_mask(mask, plt.gca())
    show_points(points_array, points_label, plt.gca())
    plt.axis('off')
    plt.show()

