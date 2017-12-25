# Finding Lane Lines on the Road

<img src="examples/laneLines.jpg" width="480" alt="Combined Image" />

## Overview
When we drive, we use our eyes to decide where to go. The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle. Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project you will detect lane lines in images using Python and OpenCV. OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps.
- Converted the images to grayscale
- Define a kernel size and apply Gaussian smoothing
- Applying Canny Edge detection to grayscale and smoothed original image
- Define region of interest (only lane) using cv2.fillPoly() (ignore everything outside region of interest)
- Define Hough transform parameters and run Hough transform on masked edge-detected image
- Draw line segments on the black image
- Draw full extend lines extrapolated from line segments
- Overlap line image with original image

In particular, to draw a single line on the left and right lanes i used the output of Hough Trasform as follow:
- Sort the Hough segment in desc way
- Extend the longest segment from the top to dowm of region of interest
- Search the 2° longest segment with opposite slope and extend as the previous step
- Draw on original image


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming is the difficul to detect the not straight lane because i'm trying to fit lines and not curves. 
Another shortcoming could be the poor generation result when there are multiple lines on the road, like in urban scenario or different sign on the road.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to use a nonlinear approch to fit the different shape of lane lines.

