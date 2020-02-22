# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:31:49 2020

@author: Sai
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage.data import load
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.measure import label, regionprops
from skimage.color import label2rgb

import numpy as np


arrow_thresh = 250  # Remove everything above this threshold from the image - useful for removing the white arrows
RGB_idx = 1 # Work on only the red, blue or green part of the image

BN = load(r'C:\Users\Sai\Documents\SpiderOak Hive\Career\Projects\Flake detection\BN flakes\BN53_S5_100X_used_34nm.jpg')
#BN = load(r'C:\Users\Sai\Documents\SpiderOak Hive\Career\Projects\Flake detection\BN flakes\BN78_S2_100X_used_51nm.jpg')
#BN = load(r'C:\Users\Sai\Documents\SpiderOak Hive\Career\Projects\Flake detection\BN flakes\BN54_S2_100X_clean_26nm_bot.jpg')

BN_red = np.array([[BN[i][j][RGB_idx] if BN[i][j][1] < arrow_thresh else 0 for i in range(len(BN))] for j in range(len(BN[0]))])

plt.figure()
plt.imshow(BN_red)
plt.colorbar()
plt.show()

thresh = threshold_otsu(BN_red)
bw = closing(BN_red > thresh, square(3))

plt.figure()
plt.imshow(bw)
#plt.colorbar()
plt.show()

label_image = label(bw, background=0)

plt.figure()
plt.imshow(label_image)
plt.colorbar()
plt.show()

regions = regionprops(label_image)

#image_label_overlay = label2rgb(label_image, image=BN_red)


fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(BN_red)

for region in regionprops(label_image):
    # take regions with large enough areas
    print(region.area)
    if region.area >= 1000:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

ax.set_axis_off()
plt.tight_layout()
plt.show()