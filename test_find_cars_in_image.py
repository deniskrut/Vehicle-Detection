import matplotlib.pyplot as plt

from find_cars_in_image import *

# Read the test image
image = mpimg.imread('test_images_2/test3.jpg')

# Find cars and draw bounding boxes on cars
window_img, _ = find_cars_in_image(image)

# Show image
plt.imshow(window_img)
plt.show()
