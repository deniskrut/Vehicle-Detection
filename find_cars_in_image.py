from collections import deque

from scipy.ndimage.measurements import label
from sklearn.externals import joblib

from extract_features import *
from helper_functions import *

find_cars_in_image_windows = None
find_cars_in_image_svc = None
find_cars_in_image_X_scaler = None


def find_cars_in_image(image, prev_hot_windows=deque([])):
    global find_cars_in_image_windows, find_cars_in_image_svc, find_cars_in_image_X_scaler

    if find_cars_in_image_svc is None:
        find_cars_in_image_svc = joblib.load('car_classifier.pkl')
    if find_cars_in_image_X_scaler is None:
        find_cars_in_image_X_scaler = joblib.load('feature_scaler.pkl')

    if find_cars_in_image_windows is None:
        image_height = image.shape[0]

        # Min and max in y to search in slide_window()
        y_start_stop_64 = [int(image_height * 0.55), int(image_height * 0.7)]
        y_start_stop_96 = [int(image_height * 0.55), int(image_height * 0.75)]
        y_start_stop_128 = [int(image_height * 0.55), int(image_height * 0.8)]
        y_start_stop_192 = [int(image_height * 0.55), int(image_height * 0.9)]
        y_start_stop_256 = [int(image_height * 0.55), None]

        windows_64 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_64,
                                  xy_window=(64, 64), xy_overlap=(0.5, 0.5))
        windows_96 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_96,
                                  xy_window=(96, 96), xy_overlap=(0.5, 0.5))
        windows_128 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_128,
                                   xy_window=(128, 128), xy_overlap=(0.5, 0.5))
        windows_192 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_192,
                                   xy_window=(192, 192), xy_overlap=(0.5, 0.5))
        windows_256 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_256,
                                   xy_window=(256, 256), xy_overlap=(0.5, 0.5))

        windows = []
        windows.extend(windows_64)
        windows.extend(windows_96)
        windows.extend(windows_128)
        windows.extend(windows_192)
        windows.extend(windows_256)

        find_cars_in_image_windows = windows

    hot_windows = search_windows_standard(image, find_cars_in_image_windows, find_cars_in_image_svc,
                                          find_cars_in_image_X_scaler)

    if len(prev_hot_windows) > 30:
        prev_hot_windows.popleft()

    prev_hot_windows.append(hot_windows)

    heatmap = np.zeros_like(image[:, :, 0]).astype(np.float)

    for cur_hot_windows in prev_hot_windows:
        add_heat(heatmap, cur_hot_windows)

    heat_threshold = len(prev_hot_windows) - 1

    heatmap = apply_threshold(heatmap, heat_threshold)

    labels = label(heatmap)

    window_img = draw_labeled_bboxes(image, labels)

    return window_img, prev_hot_windows
