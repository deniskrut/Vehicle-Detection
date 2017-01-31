import matplotlib.pyplot as plt

from find_cars_in_image import *

image = mpimg.imread('test_images_2/test3.jpg')

window_img, _ = find_cars_in_image(image)

plt.imshow(window_img)
plt.show()
