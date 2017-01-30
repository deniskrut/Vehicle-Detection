from find_cars_in_image import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('test_images/test5.jpg')

window_img = find_cars_in_image(image)

plt.imshow(window_img)