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
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/heatmap.png
[image6]: ./examples/example_output.jpg
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

I've estimated the classification performance of given set of parameters by training and testing the classifier on just HOG output. I've learned that `pixel_per_cell == 16` and `hog_channel == 0` produce almost as good classification as `pixel_per_cell == 8` and `hog_channel == "ALL"`, while taking much less time to execute. I've adjusted other parameters of the pipeline on the former, and then switched to the later.

Following parameters produced best classification accuracy, so these are my final parameters:

```
orient = 9          # HOG orientations
pix_per_cell = 8    # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
hog_feat = True     # HOG features on or off
```

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

SVM training code is located in the `train_classifier.py` file. Initially I read all the car and non-car images using `glob.glob()`.

Then I extracted all features using function `extract_features_standard` from `extract_features.py` file. That function uses spatial, histogram and HOG features to form an array of feature vectors.

Then I scaled all features using `StandardScaler` from `sklearn.preprocessing`.

Then I performed a split on training and testing data using `train_test_split` from `sklearn.model_selection`.

Then I trained my classifier using `LinearSVC` from `sklearn.svm`.

Finally, I saved scaler and SVM model to the files using `joblib`.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Observing the video I noticed that cars that are far away tend to be located toward the middle of the image height-wise, while cars that are closer to the camera can span between the middle of the image height-wise and the bottom of the image. So I located my sliding windows respectively.

In order to combat false positives I've tried to use many windows with `0.75` overlap. That gave me a huge performance hit - one video generation took 2 and a half hours. But it also did not produce desired result - windows of different sizes seem to agree where they see a car. This is probably due to some scaling used in the dataset.

TODO

So I reduced number of windows. I used `0.75` horizontal overlap for the top window of size `64x64`, for all other sizes and dimensions I used overlap of `0.5`. I also used windows of sizes `96x96`, `128x128`, `160x160`, `192x192`, `224x224` and `256x256`

TODO

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to try to minimize false positives and reliably detect cars?

In order to minimize false positives I've tried threshing. I've built a heat map for all found cars, and made a threshold to cut off false positives. However this did not seem enough - there were some false negatives having same threshold as false positives.

Then I've tried using more search windows and higher threshold. That did not helped, as described in previous section.

After this I shifted my attention to filtering false positives on video pipeline level, which is described below.

TODO

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

TODO

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then threshold that map to identify vehicle positions.  I then used blob detection in Sci-kit Image (Determinant of a Hessian [`skimage.feature.blob_doh()`](http://scikit-image.org/docs/dev/auto_examples/plot_blob.html) worked best for me) to identify individual blobs in the heatmap and then determined the extent of each blob using [`skimage.morphology.watershed()`](http://scikit-image.org/docs/dev/auto_examples/plot_watershed.html). I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap and bounding boxes overlaid on a frame of video:

![alt text][image5]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

TODO

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

