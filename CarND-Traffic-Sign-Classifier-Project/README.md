# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/example_dataset.jpg "Dataset"
[image3]: ./examples/grayscale.jpg "Normalized"
[image4]: ./examples/web.jpg "Web"
[image5]: ./examples/softmax.jpg "Softmax"

## Rubric Points

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* Dataset available here: [dataset.zip](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)
* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data are distributed.
Each bar represents one class (= traffic sign) and how many samples there are for that class. The mapping between the traffic sign names and the class ID can be found here: signnames.csv

![alt text][image1]

Some examples of train 32x323x dataset

![alt text][image2]

### Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale be in YCrCb color space and using only the first channel /Y-channel). In addiction it was applied a normalization of the pixels, reducing the values between 0 and 1 .Here is an example of a traffic sign image after the normalizzation:

![alt text][image3]

Reducing the amount of input data, reducing the channels, the train step of the model is significantly faster. 

#### Model Architecture
 
 I use a convolutional neuronal network to classify the traffic signs. The input of the network is an 32x32x1 image and the output is the probabilty of each of the 43 possible traffic signs.
 
 My final model consisted of the following layers:

| Layer         		|     Description	        					| Input |Output| 
|:---------------------:|:---------------------------------------------:| :----:|:-----:|
| Convolution 5x5     	| 1x1 stride, valid padding, RELU activation 	|**32x32x1**|28x28x6|
| Max pooling			| 2x2 stride, 2x2 window						|28x28x6|14x14x6|
| Convolution 5x5 	    | 1x1 stride, valid padding, RELU activation 	|14x14x6|10x10x16|
| Max pooling			| 2x2 stride, 2x2 window	   					|10x10x16|5x5x16|
| Convolution 6x6 		| 1x1 stride, valid padding, RELU activation    |14x14x6|10x10x486|
| Flatten				| 3 dimensions -> 1 dimension					|400| 486|
| Fully Connected | connect every neuron from layer above			|886|84|
| Fully Connected | output = number of traffic signs in data set	|84|**43**|


#### Model Training
I used my GPU (NVIDA GeForce GT 630 M) to tran the mode. It is slightly but ompared to the cpu, the training was about 3.3 times faster.

* EPOCHS = 20
* BATCH_SIZE = 128
* SIGMA = 0.1
* OPIMIZER: AdamOptimizer with learning rate = 0.001

My results after training the model:
* Training Accuracy = 100% 
* Validation Accuracy = 91.5%
* Test Accuracy = 90.08%

My first approach was LeNet-5 shown in the udacity classroom. Despite it was a good starting point i got low validation accuracy about 85%. Therefore i decided to modify the network using Sermanet/LeCunn traffic sign classification paper.
Adding one more convolutional layer, the newtwork got better results. Training for more than 20 epochs do not increase the validation accuracy instead, instead it was going to decrease. 
 

### Test a Model on New Images

Here are 6 examples I collected.

![alt text][image4] 

The first image might be difficult to classify because ...

| Image			        |     Prediction		| 
|:---------------------:|:---------------------:| 
| Speed limit (30km/h)  | Speed limit (30km/h)  | 
| Priority road   		| Children crossing 	|
| Yield			| Bicycle crossing				|
| Stop		| Stop					|
| No entry		| No entry  |
| Keep right | Keep right |


The model was able to correctly guess 5 of the 6 traffic signs = **83.3 %** 
#### Softmax Probabilities
The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

![alt text][image5] 

For the second image ... 


