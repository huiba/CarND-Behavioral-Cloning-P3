# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I designed a small architecture and also used the architecture from Nvidia paper with some small modifications.
The self designed model is showed below. 

|      Layer      |                 Description                  |
| :-------------: | :------------------------------------------: |
|      Input      |              90x320x3 RGB image              |
| Convolution 3x3 | 1x1 stride, valid padding, outputs 80x318x8  |
|      RELU       |                                              |
| Convolution 3x3 | 1x1 stride, valid padding, outputs 80x316x16 |
|      RELU       |                                              |
|   Max pooling   |        2x2 stride,  outputs 43x158x16        |
| Convolution 3x3 | 1x1 stride, valid padding, outputs 41x156x32 |
|      RELU       |                                              |
| Convolution 3x3 | 1x1 stride, same padding, outputs 39x154x32  |
|      RELU       |                                              |
|   Max pooling   |        2x2 stride,  outputs 19x77x32         |
| Convolution 3x3 | 1x1 stride, valid padding, outputs 17x75x64  |
|      RELU       |                                              |
| Convolution 3x3 |  1x1 stride, same padding, outputs 15x73x64  |
|      RELU       |                                              |
|   Max pooling   |         2x2 stride,  outputs 7x36x64         |
| Fully connected |                 outputs 128                  |
|     Dropout     |                  prob: 0.3                   |
| Fully connected |                  outputs 32                  |
|     Dropout     |                  prob: 0.3                   |
| Fully connected |                  outputs 1                   |
#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

I also tried the l2 regularization, but it did not work well. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used mean squre error as loss function and an adam optimizer. The default learning rate (0.01) was not able the train a working network (The network ended up with outputing a constant value). I decreased the learning rate to 0.001. And trained the models for 10 epoches.

#### 4. Appropriate training data

In the final model, I just used the data provided by Udacity. I did collect many data myself. Also, I followed the insturction to get all the data, normal driving for 2 laps, recover driving from the road side, and smooth curve driving. But with all these network, the network performs bad, it can not even run a whole lap even with slow speed. 

To preprocess the data, I just crop the input images and normalize the them. 

#### 5. Results

As mentioned above, I trained both of the networks only with udacity data. 
I use Adam optimizer with leraning rate 0.001, and dropout to prevent overfitting.

With the self designed model with the speed 10, the model can finish the track but with some collision with the side. After the collsion, the car was able to recover to run on the track with a great and sharp turn. I found that interesting. 

The nvidia model works just fine. Even with a high speed 20, it can also finish the whole track. 

### Disscusion

* With the default learning rate (0.01). The models are trained still with dropping losses. But at the end the networks will just output the same value, no matter how the input changes. It seems the quality of the data has a huge impact on the training.

* I did collected some training data by driving the car. And I also carefully tuned all the other hyper parameters, but did not get a resonable results. Maybe I just did not drive so good, or too fast?

* The L2 regularization just did not make any effect on the training. 

