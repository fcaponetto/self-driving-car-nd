## Advanced Lane Finding

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[img01]: ./readme_images/corners.jpg "Chessboard Calibration"
[img02]: ./readme_images/undistort_chessboard.jpg "Undistort Chessboard"
[img03]: ./readme_images/undistort_car.jpg "Undistort Car"
[img04]: ./readme_images/warped_lane.jpg "Warped Lane"
[img05]: ./readme_images/binary_threashold_test_1.jpg "Binary Threadhold test 1"
[img06]: ./readme_images/binary_threashold_test_2.jpg "Binary Threadhold test 2"
[img07]: ./readme_images/fill_lane.jpg "Fill Lane"
[vid01]: ./readme_images/project_video_output.gif "Output Gif"
[vid02]: ./readme_images/challenge_video_output.gif "Output Challenge"

### Step1: Camera Calibration

The OpenCV functions *findChessboardCorners* and *calibrateCamera* are two useful function provided for image calibration. Taking different images from different angles with the same camera, the pixel locations of the internal chessboard corners determined by *findChessboardCorners*, are fed to *calibrateCamera* which returns camera calibration and distortion coefficients. 

![alt text][img01]

These can then be used by the OpenCV *undistort* function to undo the effects of lent distortion .

![alt text][img02]

![alt text][img03]

### Step 2: Perspective Transform

With this step we wanto to transform the undistorted image to a "birds eye view" of the road which focuses only on the lane lines.
To achieve the perspective transformation OpenCV provide the functions *getPerspectiveTransform* and *warpPerspective* which take a matrix of four source points on the undistorted image and remaps them to four destination points on the warped image.

[img04]: ./readme_images/warped_lane.jpg "Warped Lane"

### Step 3: Binary Thresholds

The goal of this step is to starting from the colored warped image, trasporming it wiht different color spaces and create binary thresholded images which highlight only the lane lines and ignore everything else.
I tried different combination of color space, in paticular,

* The S Channel from the HLS color space, did a fairly good job of identifying both the white and yellow lane lines.
* The L Channel from the HLS color space, did an almost perfect job of picking up the white lane lines, but ignore the yellow lines.
* The B channel from the Lab color space, did a better job than the S channel in identifying the yellow lines, but ignore the white lines.

![alt text][img05]

![alt text][img06]

At the end, i decided to use the combination of the best yellow and with line, that's L channel + B channel.

### Step 4: Fit Lane

The function *fit_lane* identify lane lines and fit a second order polynomial to both right and left lane lines:
* First it computes a histogram of the bottom half of the image and finds the bottom-most x position (or "base") of the left and right lane lines.
* Identifying all non zero pixels around histogram peaks using the numpy function numpy.nonzero().
* The function then identifies 9 windows from which to identify lane pixels, each one centered on the midpoint of the pixels from the window below

After using the polynomials I was able to calculate the position of the vehicle with respect to center:

* Calculated the distance from center
* The radius of curvature is based upon [this website](http://www.intmath.com/applications-differentiation/8-radius-curvature.php)

![alt text][img07]

### Video Processing Pipeline

The final step was to expand the pipeline to process videos frame-by-frame, to simulate what it would be like to process an image stream in real time on an actual vehicle.

|Project Video|
|-------------|
|![alt text][vid01]|

### Limitations 
The problems I encountered were almost exclusively due to lighting conditions, shadows. It shows a road in basically ideal conditions, with fairly distinct lane lines, and on a clear day, with some problem when the line are not well defined or appears some shadows on the lane.
