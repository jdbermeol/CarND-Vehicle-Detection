**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car.png
[image2]: ./output_images/no-car.png
[image3]: ./output_images/hog.png
[image4]: ./output_images/Final.png
[image5]: ./output_images/heatmap.png
[image6]: ./output_images/labelmap.png
[image7]: ./output_images/rectangles.png
[image8]: ./output_images/rectangles2.png

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Code cells are in Train vehicle detector model.ipynb, cells are from 1 to 20.

Cells 1 and 2 define basic imports and a unify way to read images from files across the code.

Then I started by reading in all the `vehicle` and `non-vehicle` images, cell 3.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes: 

![alt text][image1] ![alt text][image2]

Cell 4 sfuggles the data, cell 5 display class size to evaluate class invalance. Cell 6 Split on train and test dataset.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). Here is an example using the `HLS` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`: 

![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters , number of bins for color histogram, number of orientations, and cell size. At the end the feature selection was done measuring the f1-score(robust measure against class imbalance) of a Linear SVM with a 3-fold cross validation.

I discarded RGB color space, for its undesirable properties under changing light conditions. I deside to use HSV sapce for color features and HLS space for shape features.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM doing a 3-fold cross validation evaluating different values for parameter C. Code goes from cell 21 to 28. Final test accuracy was 97% and 0.97 f1-score.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Code is in Detect vehicles.ipynb. I decided to search random window positions at random scales all over the image.

![alt text][image7]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales (1, 1.5, 2, 3.5) on different windows using HLS 3-channel HOG features plus HSV histograms of color in the feature vector, which provided a nice result.

![alt text][image8]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output_images/test_video_out.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap:

![alt text][image6]

### Here the resulting bounding boxes:
TODO

![alt text][image4]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Current implementation has several potential issues.

1. Even though accuracy is around 97%, it means model misclassify several windows, and a threshold strategy is not robust enough, it would be good to find a more accurate(around 99%) model.

2. Sliding window search resolution is not high enough, small cars are going to have issues as overlapping is not okay. Increase resolution would help overall accuracy.

3. Each frame analysis is expensive, implement and strategy to take the previous frames detections would help to improve performance and efficiency.

4. Classification model(Linear SVM) requires a small dataset to be trained. This is an issue when new samples not similar to the training dataset happens. A model that can learn from a huge dataset would be better.

5. Detection is slow, and several things can be parallelized and it should. 

