import cv2
import matplotlib.pyplot as plt


def showImgByPath(path):
  """Show the image for filepath <path>"""
  image = cv2.imread(path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  plt.figure()
  plt.imshow(image)

def readImgByPath(path):
  """Return the image for filepath <path> """
  image = cv2.imread(path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image

def showImg(image, title=""):
  """Show the image, which content is in image."""
  plt.figure()
  plt.imshow(image)
  if title!="":
    plt.title(title)
  plt.show()

def showTwoImgs(img1, img2):
  """Show both images, side by side."""
  plt.subplot(1,2,1);
  plt.imshow(img1);
  plt.subplot(1,2,2);
  plt.imshow(img2);
  plt.show()

def showThreeImgs(image1, image2, image_to_mask, masks = None, titles = None):
  # Create a figure with three subplots
  fig, axes = plt.subplots(1, 3, figsize=(15, 5))
  # Display the first image in the first subplot
  axes[0].imshow(image1)
  
  
  # Display the second image in the second subplot
  axes[1].imshow(image2)
  
  
  # Display the third image in the third subplot
  axes[2].imshow(image_to_mask)
  

  if masks:
    if len(masks) == 0:
      return
    img = np.ones((masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for mask in masks:
        m = mask['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    axes[2].imshow(img)
  if titles:
    axes[0].set_title(titles[0])
    axes[1].set_title(titles[1])
    axes[2].set_title(titles[2])
  
  # Adjust spacing between subplots
  plt.tight_layout()
  
  # Show the figure
  plt.show()
