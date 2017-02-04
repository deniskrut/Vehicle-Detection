import math

from scipy.ndimage.measurements import label
from sklearn.externals import joblib

from extract_features import *
from helper_functions import *

# Cache for windows, classifier and scaler
find_cars_in_image_windows = None
find_cars_in_image_svc = None
find_cars_in_image_X_scaler = None


def find_cars_in_image(image, prev_hot_windows=None):
    # Load caches
    global find_cars_in_image_windows, find_cars_in_image_svc, find_cars_in_image_X_scaler

    # Rehydrate caches if needed
    if find_cars_in_image_svc is None:
        # Load classifier from file
        find_cars_in_image_svc = joblib.load('car_classifier.pkl')

    if find_cars_in_image_X_scaler is None:
        # Load scaler from file
        find_cars_in_image_X_scaler = joblib.load('feature_scaler.pkl')

    if find_cars_in_image_windows is None:
        image_height = image.shape[0]

        # Min and max in y to search in slide_window()
        y_start_stop_64 = [int(0.525 * image_height), int(0.7 * image_height)]
        y_start_stop_128 = [int(0.525 * image_height), int(0.75 * image_height)]
        y_start_stop_192 = [int(0.525 * image_height), int(0.8 * image_height)]
        y_start_stop_256 = [int(0.525 * image_height), int(0.9 * image_height)]
        y_start_stop_320 = [int(0.525 * image_height), int(1 * image_height)]

        # Obtain windows given set of parameters
        windows_64 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_64,
                                  xy_window=(64, 64), xy_overlap=(0.85, 0.5))
        windows_128 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_128,
                                   xy_window=(128, 128), xy_overlap=(0.65, 0.5))
        windows_192 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_192,
                                   xy_window=(192, 192), xy_overlap=(0.6, 0.5))
        windows_256 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_256,
                                   xy_window=(256, 256), xy_overlap=(0.6, 0.5))
        windows_320 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_320,
                                   xy_window=(320, 320), xy_overlap=(0.6, 0.5))

        # Store all windows
        windows = []
        windows.extend(windows_64)
        windows.extend(windows_128)
        windows.extend(windows_192)
        windows.extend(windows_256)
        windows.extend(windows_320)

        # Store windows in cache
        find_cars_in_image_windows = windows

    # Find windows that actually have vehicles
    hot_windows = search_windows_standard(image, find_cars_in_image_windows, find_cars_in_image_svc,
                                          find_cars_in_image_X_scaler)

    # Allocate a heatmap for windows having vehicles
    heatmap = np.zeros_like(image[:, :, 0]).astype(np.float)

    # Add heat for found windows that have a car
    add_heat(heatmap, hot_windows)
    # In video we want to rely on many frames reporting a car, rather then one being really confident that it has a car
    # For that reason we are using square root of the heatmap.
    heatmap = np.sqrt(heatmap)

    # Heat threshold to filter false positives
    heat_threshold = 1

    # If we a processing a video, we might have hot windows information from previous frames
    if prev_hot_windows is not None:
        # We look back 15 frames.
        # It is significant number, but I use heat map cooling to negate negative effect of too many look back frames.
        look_back_count = 15

        # Iterate through each historical hot window
        for index, cur_hot_windows in enumerate(prev_hot_windows):
            # Cooling function: sine loss of heat to give newer frames higher value
            scaler = math.sin((math.pi / 2) * (index / len(prev_hot_windows)))

            # Allocate new heat map
            current_heatmap = np.zeros_like(image[:, :, 0]).astype(np.float)
            # Add heat to the heat map
            add_heat(current_heatmap, cur_hot_windows)
            # Take square root of heat map to reduce potential influence of one frame
            # Also apply scaler after the sqrt to keep the descending curve (so all numbers are >= 1 for sqrt)
            current_heatmap = np.sqrt(current_heatmap) * scaler
            # Add current heat map to heat map of this frame
            heatmap += current_heatmap

        # If we accumulated too many frames, remove one
        if len(prev_hot_windows) > look_back_count:
            prev_hot_windows.popleft()

        # Append current frame to the list of previous hot frames
        prev_hot_windows.append(hot_windows)

        # Adjust threshold to accommodate added heat
        heat_threshold *= look_back_count

    # Apply threshold to the heatmap
    heatmap_thresholded = apply_threshold(heatmap, heat_threshold)

    # Obtain bounding boxes for interconnected heatmap areas
    labels = label(heatmap_thresholded)

    # Draw labeled bounding boxes based on labels
    window_img = draw_labeled_bboxes(image, labels)

    # Return resulting image and previous hot windows
    return window_img, prev_hot_windows
