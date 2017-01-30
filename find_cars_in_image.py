from sklearn.externals import joblib
from extract_features import *
from helper_functions import *
import matplotlib.pyplot as plt

y_start_stop = [300, None] # Min and max in y to search in slide_window()

svc = joblib.load('car_classifier.pkl')
X_scaler = joblib.load('feature_scaler.pkl')

image = mpimg.imread('test_images/test5.jpg')
draw_image = np.copy(image)

windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))

hot_windows = search_windows_standard(image, windows, svc, X_scaler)

window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

plt.imshow(window_img)