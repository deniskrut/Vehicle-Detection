from sklearn.externals import joblib
from extract_features import *
from helper_functions import *
from scipy.ndimage.measurements import label

def find_cars_in_image(image):
    image_height = image.shape[0]

    # Min and max in y to search in slide_window()
    y_start_stop_64 = [int(image_height * 0.55), int(image_height * 0.7)]
    y_start_stop_128 = [int(image_height * 0.55), int(image_height * 0.8)]
    y_start_stop_256 = [int(image_height * 0.55), None]

    svc = joblib.load('car_classifier.pkl')
    X_scaler = joblib.load('feature_scaler.pkl')

    draw_image = np.copy(image)

    windows_64 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_64,
                        xy_window=(64, 64), xy_overlap=(0.5, 0.5))
    windows_128 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_128,
                        xy_window=(128, 128), xy_overlap=(0.5, 0.5))
    windows_256 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_256,
                        xy_window=(256, 256), xy_overlap=(0.5, 0.5))

    windows = []
    windows.extend(windows_64)
    windows.extend(windows_128)
    windows.extend(windows_256)

    hot_windows = search_windows_standard(image, windows, svc, X_scaler)

    heatmap = np.zeros_like(image[:, :, 0]).astype(np.float)

    add_heat(heatmap, hot_windows)

    heatmap = apply_threshold(heatmap, 1)

    labels = label(heatmap)

    window_img = draw_labeled_bboxes(image, labels)

    return window_img