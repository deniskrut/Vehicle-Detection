#Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1a]: ./examples/car.png
[image1b]: ./examples/not_car.png
[image2a]: ./examples/yuv_0.png
[image2b]: ./examples/yuv_1.png
[image2c]: ./examples/yuv_2.png
[image3]: ./examples/sliding_windows_1.png
[image4]: ./examples/sliding_windows_2.png
[image5]: ./examples/heatmap.png
[video1]: ./project_video_solution.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is performed by the `extract_features_standard` function, located in the `extract_features.py` file.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Car][image1a]
![Not car][image1b]

Then for each image I've defined set of parameters to be used in HOG extraction. This step can be seen in the function `single_img_features_standard`, located in the `extract_features.py` file.

Then I converted images to the `YUV` color space in the function `single_img_features`, located in the `helper_functions.py` file.

Then I used `skimage.feature.hog()` function to extract a feature vector from each channel using function `get_hog_features` located in `helper_functions.py` file.

####2. Explain how you settled on your final choice of HOG parameters.

I explored different color spaces and different `skimage.feature.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.feature.hog()` output looks like. Here is an example using the `YUV` color space and HOG parameters of `orientations = 9`, `pixels_per_cell = (8, 8)` and `cells_per_block = (2, 2)` on each channel of car image shown above.

![YUV channel 0][image2a]
![YUV channel 1][image2b]
![YUV channel 2][image2c]

I've estimated the classification performance of given set of parameters by training and testing the classifier on just HOG output. I've learned that `pixel_per_cell == 16` and `hog_channel == 0` produce almost as good classification as `pixel_per_cell == 8` and `hog_channel == "ALL"` (98% vs 99% accuracy), while taking much less time to execute. I've adjusted other parameters of the pipeline on the former, and then switched to the later.

Following parameters produced best classification accuracy, so these are my final parameters:

```python
orient = 9          # HOG orientations
pix_per_cell = 8    # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
hog_feat = True     # HOG features on or off
```

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

SVM training code is located in the `train_classifier.py` file. Initially I read all the car and non-car images using `glob.glob()`.

Then I extracted all features using function `extract_features_standard` from `extract_features.py` file. That function uses binned colors, histogram and HOG features to form an array of feature vectors.

Then I scaled all features using `StandardScaler` from `sklearn.preprocessing`.

Then I performed a split on training and testing data using `train_test_split` from `sklearn.model_selection`.

Then I trained my classifier using `LinearSVC` from `sklearn.svm`.

Finally, I saved scaler and SVM model to the files using `joblib`.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Observing the video I noticed that cars that are far away tend to be located toward the middle of the image height-wise, while cars that are closer to the camera can span between the middle of the image height-wise and the bottom of the image. So I located my sliding windows respectively.

I've obtained a few images from the video where my pipeline had most problems, and tried to optimize those by changing the sliding windows.

I've started with sizes `64x64`, `96x96`, `128x128`, `160x160`, `192x192`, `224x224` and `256x256` and overlap `0.5`. That produced good results, but it missed some smallest and largest cars. I was able to improve it by increasing horizontal overlap to `0.75` for smallest windows, and adding size `320x320`. However at that point execution time started to suffer - it could take two and a half hours to produce the project view.

As a next step, I've tried to reduce number of window and overlaps to improve execution time. I've noticed that bigger windows needed smaller overlaps to perform well, and that I can also reduce number of different sizes without loosing detection quality. After this optimization, when combined with "lighter" feature extraction option described above, pipeline can produce couple of frames a second.

Code for this procedure is located in the `find_cars_in_image` function, `file_cars_in_image.py` file. This function uses caching for SVM, Scaler and sliding windows to improve performance. On the first execution of the function I read SVM and scaler from files and generate sliding windows. In order to generate sliding windows I assign sizes, search areas and overlaps and supply it to `slide_window` function located in `helper_functions.py`. Here is what these parameters look like:

```python
import matplotlib.image as mpimg
from helper_functions import *

image = mpimg.imread('test_images_2/test3.jpg')
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
```

`slide_window` function outputs bounding boxes for windows with corresponding parameters.

Next I search for windows using `search_windows_standard` function form `extract_features.py` file. This function for each window obtains a scaled image, extract features for that image, and forms an array of all features extracted for all windows. Then, using SVM, function performs prediction. Array of windows where cars are predicted is returned from the function.

Here is an example of the array of windows, each of which recognized a car:

![Sliding windows][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to try to minimize false positives and reliably detect cars?

In order to minimize false positives I've tried heat maps and threshing. I've built a heat map for all found cars, and made a threshold to cut off false positives. However this did not seem enough - there were some false negatives having same threshold as false positives.

Next I've tried to improve detection quality to have less false positives. I've checked standalone performance of each component forming the feature vector: binned color features, histogram and HOG. By combining best results I was able to achieve 99% accuracy, while before I was at 98%. However that did not help to get rid of false positives, and also increased time needed to process video significantly.

Next I've tried to use many windows with `0.75` overlap. That gave me a huge performance hit - one video generation took two and a half hours. But it also did not produce desired result - windows of different sizes seem to agree where they see a car. This is probably due to some scaling used in the dataset.

After this I shifted my attention to filtering false positives on video pipeline level, which is described below.

You can find code for heatmap and threshing in the `find_cars_in_image` function in the `find_cars_in_image.py` file. `add_heat` function from `helper_functions.py` adds 1 for every area marked by bounding box corresponding to found vehicle. Then `apply_threshold` function from same file zeros out all parts of heat map that are below given threshold (in my case, `1`). `label` function from `scipy.ndimage.measurements` finds bounding boxes for the heat map. Finally, `draw_labeled_bboxes` function from `helper_functions.cpp` draws bounding boxes on the image.

Here is an example of heatmap with bounding boxes:

![Sliding windows and heatmap][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To filter false positives for video I used hot window history from previous frames. I took last 30 frames and produced weighted average for square roots of heat maps. Result of this operation was used as a new heat map.

For a weighted average I used sine function, since I wanted more recent frames to have greater influence then older ones. This helped me to smooth the bounding box jarring and reject false positives that are only present on a few frames, but not the rest of them.

In order to mitigate influence of one frame on the final outcome, I have applied a square root to the feat maps. This assures that there is no one super-confident frame that will introduce false positive, even though there is no detections in that area in consecutive frames.

Code for this feature can be found in the `find_cars_in_image` function, file `find_cars_in_image.py`.

```python
import math
from helper_functions import *

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
```

Here's an example result showing the heatmap and bounding boxes overlaid on a frame of video:

![Hetmap in video frame][image5]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Detection accuracy was not best for the video, so I had to compensate by handpicking parameters and extensively using history. Handpicking parameters may fail in real world, and history needs multiple frames to kick in. Also, each frame takes at least half a second to produce on my machine, which is not acceptable in production.

In real world U-net seem to be a great idea. They are fast and very accurate.

Another idea is to augment the data we have by adding and removing light, stretching, etc. This should make it work better on the video. Finally, I could try using different data set in addition to the current ones. [Udacity annotated driving data set](https://github.com/udacity/self-driving-car/tree/master/annotations) seem to be a better match in terms of lighting conditions at least.
