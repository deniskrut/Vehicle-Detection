from scipy.ndimage.measurements import label
from sklearn.externals import joblib

from extract_features import *
from helper_functions import *

find_cars_in_image_windows = None
find_cars_in_image_svc = None
find_cars_in_image_X_scaler = None


def find_cars_in_image(image, prev_hot_windows=None):
    global find_cars_in_image_windows, find_cars_in_image_svc, find_cars_in_image_X_scaler

    if find_cars_in_image_svc is None:
        find_cars_in_image_svc = joblib.load('car_classifier.pkl')
    if find_cars_in_image_X_scaler is None:
        find_cars_in_image_X_scaler = joblib.load('feature_scaler.pkl')

    if find_cars_in_image_windows is None:
        image_height = image.shape[0]

        # Min and max in y to search in slide_window()
        y_start_stop_64 = [int(0.525 * image_height), int(0.7 * image_height)]
        y_start_stop_96 = [int(0.525 * image_height), int(0.725 * image_height)]
        y_start_stop_128 = [int(0.525 * image_height), int(0.75 * image_height)]
        y_start_stop_160 = [int(0.525 * image_height), int(0.775 * image_height)]
        y_start_stop_192 = [int(0.525 * image_height), int(0.8 * image_height)]
        y_start_stop_224 = [int(0.525 * image_height), int(0.825 * image_height)]
        y_start_stop_256 = [int(0.525 * image_height), int(0.85 * image_height)]

        windows_64 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_64,
                                  xy_window=(64, 64), xy_overlap=(0.75, 0.5))
        windows_96 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_96,
                                  xy_window=(96, 96), xy_overlap=(0.5, 0.5))
        windows_128 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_128,
                                   xy_window=(128, 128), xy_overlap=(0.5, 0.5))
        windows_160 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_160,
                                   xy_window=(160, 160), xy_overlap=(0.5, 0.5))
        windows_192 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_192,
                                   xy_window=(192, 192), xy_overlap=(0.5, 0.5))
        windows_224 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_224,
                                   xy_window=(224, 224), xy_overlap=(0.5, 0.5))
        windows_256 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_256,
                                   xy_window=(256, 256), xy_overlap=(0.5, 0.5))

        windows = []
        windows.extend(windows_64)
        windows.extend(windows_96)
        windows.extend(windows_128)
        windows.extend(windows_160)
        windows.extend(windows_192)
        windows.extend(windows_224)
        windows.extend(windows_256)

        find_cars_in_image_windows = windows

    hot_windows = search_windows_standard(image, find_cars_in_image_windows, find_cars_in_image_svc,
                                          find_cars_in_image_X_scaler)

    heatmap = np.zeros_like(image[:, :, 0]).astype(np.float)

    add_heat(heatmap, hot_windows)
    heatmap = np.sqrt(heatmap)

    heat_threshold = 1

    if prev_hot_windows is not None:
        look_back_count = 60

        for index, cur_hot_windows in enumerate(prev_hot_windows):
            amount = index / len(prev_hot_windows)
            add_heat(heatmap, cur_hot_windows, amount)
            heatmap = np.sqrt(heatmap)

        if len(prev_hot_windows) > look_back_count:
            prev_hot_windows.popleft()

        prev_hot_windows.append(hot_windows)

        heat_threshold *= (look_back_count / 35)

    heatmap = apply_threshold(heatmap, heat_threshold)

    labels = label(heatmap)

    window_img = draw_labeled_bboxes(image, labels)

    return window_img, prev_hot_windows
