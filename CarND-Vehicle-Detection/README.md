# Vehicle Detection
The main goal of the project is to create a software pipeline to identify vehicles in a video from a front-facing camera on a car.

[img01]: ./output_images/output_images.jpg "Random images"
[img02]: ./output_images/hog_transform.jpg "HOG"
[img03]: ./output_images/find_car.jpg "Find Cars"
[img04]: ./output_images/heatmap.jpg "Heatmap"
[img05]: ./output_images/heatmap_threashold.jpg "Heatmap Threashold"
[img06]: ./output_images/final.jpg "Final"
[vid01]: ./project_video_output.gif "Output Gif"

|Project Video|
|-------------|
|![alt text][vid01]|

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

### Histogram of Oriented Gradients (HOG)

The figure below shows a random sample of vehicle and non-vehicle images from both classes of the dataset: 

![alt text][img01]

Using the code for extracting HOG features from an image, defined by the method get_hog_features, we can get for and image test the figure below that shows a comparison of a car image and its associated histogram of oriented gradients:

![alt text][img02]

The method extract_features with some parameters extract features for the entire dataset. These feature sets are combined and a label vector is defined (1 for cars, 0 for non-cars).

Final parameter for feature extraction:
``
YCrCb colorspace, 9 orientations, 8 pixels per cell, 2 cells per block, and ALL channels of the colorspace.
``

### Sliding Window Searching

I used the method find_cars from the lesson materials. The method combines HOG feature extraction with a sliding window search. It allows to search a car in a desired region of the frame with a desired window size and the HOG features are extracted for the entire image (or a selected portion of it).

![alt text][img03]

The classifier successfully finds cars on the test images. However, there is a false positive example, therefore i applied the heatmap filter with threasholding:

![alt text][img04]

![alt text][img05]

And the final detection area: 

![alt text][img06]

### Video Implementation
The video implementation consist of pipeline of previous approch for each video frame:
* find cars using sliding window for subset of the frame
* apply heatmap threashold
* print the filtered boxes

### Discussion 

The main useful approach I used is HOG subsampling because of which the video processing time is much reduced and pipeline is efficient. Anyway i faced with the followig limitations:

* The algorithm may fail in case of difficult light conditions
* The algorithm has some problems in case of car overlaps to anothers
