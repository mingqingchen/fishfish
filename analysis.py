import matplotlib.image as mpimg

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb


rgb_image=mpimg.imread('./data/fish_sample.jpeg')

image = np.mean(rgb_image, axis=2)

# apply threshold
thresh = threshold_otsu(image)

bw = image < thresh
# bw = closing(image > thresh, square(3))
# remove artifacts connected to image border
cleared = clear_border(bw)

# label image regions
label_image = label(cleared)


fig, ax = plt.subplots(figsize=(10, 6))

ax.imshow(rgb_image)

for region in regionprops(label_image):
    # take regions with large enough areas
    # draw rectangle around segmented coins
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=0.5)
    ax.add_patch(rect)

plt.show()