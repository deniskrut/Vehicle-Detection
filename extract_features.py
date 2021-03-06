import matplotlib.image as mpimg

from helper_functions import *


# Obtains a feature vector using preselected set of parameters
def single_img_features_standard(image):
    # Standartize images to be uint8 data type
    if isinstance(image[0][0][0], np.float32):
        image = np.uint8(image * 255)

    color_space = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

    spatial_size = (32, 32)  # Spatial binning dimensions
    spatial_feat = True  # Spatial features on or off

    hist_bins = 64  # Number of histogram bins
    hist_feat = True  # Histogram features on or off

    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
    hog_feat = True  # HOG features on or off

    return single_img_features(image, color_space=color_space,
                               spatial_size=spatial_size, hist_bins=hist_bins,
                               orient=orient, pix_per_cell=pix_per_cell,
                               cell_per_block=cell_per_block,
                               hog_channel=hog_channel, spatial_feat=spatial_feat,
                               hist_feat=hist_feat, hog_feat=hog_feat)


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features_standard(imgs):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)

        file_features = single_img_features_standard(image)

        features.append(file_features)
    # Return list of feature vectors
    return features


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows_standard(img, windows, clf, scaler):
    # 1) Create an empty array for feature vectors
    all_features = []

    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features_standard(test_img)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Put extracted features in the feature vector
        all_features.extend(test_features)

    # 7) Using a classifier predict classes for feature vectors
    predictions = clf.predict(all_features)

    # 8) Convert predictions array of windows that have vehicles
    on_windows = [x for i, x in enumerate(windows) if predictions[i] == 1]

    # 9) Return windows for positive detections
    return on_windows
