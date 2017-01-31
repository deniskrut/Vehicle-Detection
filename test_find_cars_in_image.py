import matplotlib.pyplot as plt

from find_cars_in_image import *

image = mpimg.imread('test_images/test6.jpg')

window_img, _ = find_cars_in_image(image)

plt.imshow(window_img)
plt.show()
