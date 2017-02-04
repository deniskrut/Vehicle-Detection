import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from helper_functions import *

# Read test image
image = mpimg.imread('./examples/car.png')
# Convert image to YUV color space
yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

# Define HOG parameters
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 2

# Obtain hog features image
_, window_img = get_hog_features(yuv_image[:, :, hog_channel], orient,
                                 pix_per_cell, cell_per_block, vis=True, feature_vec=False)

# Show HOG features image
plt.imshow(window_img)
plt.show()
