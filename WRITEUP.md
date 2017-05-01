# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/unbalanced.png "Raw training data"
[image2]: ./examples/stratified.png "Stratified training data"
[image3]: ./examples/inputs.png "Input images"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* WRITEUP.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the nvidia architecture: 3 convolutional 5x5 layers with strides of 2 followed by 2 3x2 convolutional layers each with relu activations. These are followed by linear layers of 100,50,10,1 neurons. (model.py 115-136)
 
As the simulation input is "easy" I also trained a tiny architecture (tiny_model.py) with success that processed a subsampled input image (32x16) and only had a single 3x3 convolution layer with max pooling with relu activation. Although it learned to drive the test track I kept the nvidia model as it would have more capacity to learn real life complexity. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 121). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 161). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 157).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I trained the nvidia model on AWS GPUs and the tiny model on CPU. As the input for the tiny network is quite small it was much faster to prepare all the inputs first and then train. For the larger nvidia network on the GPU I used the generator approach.

Seeing the MSE going down during the training did not indicate how good the model was in actual driving, I always had to test in the simulator how far the model could drive.

It seemed that the training data mattered a lot more than the model architecture, see point 3 for the details on preparing the data.


#### 2. Final Model Architecture


````
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           cropping2d_input_2[0][0]         
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 65, 320, 3)    0           cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
____________________________________________________________________________________________________
````



#### 3. Creation of the Training Set & Training Process

First I recorded two laps of my manual driving for both driving directions. For every frame I used all 3 camera views and also the 3 flipped images, so 6 in total. I added steering corrections for the left and right images, for example: 

![alt text][image3]

The trained behaviour looked OK, but was not capable of completing the lap, the car left the track at one curve or the other. To counter this I recorded a few manual examples of more drastic steering to stay on track. This made the driving more unstable, did not solve the problem in itself.
 
I then went back to the provided udacity training data and looked at how many examples there were for the different steering angles:
 
 ![alt text][image1]

Most of the training examples are from the straight sections with near 0 angles, the data is quite biased towards driving straight. I implemented stratified sampling to have equal number (500) of examples from each of the 10 bins of steering angle ranges.

This made the car steer left to right to left, etc all the time as now it saw too much of the extreme steering examples. I reduced the more extrame angles to 50 examples and the more common ones (including 0) to 500. With this I finally managed to get a stable nice lap around track 1.

 ![alt text][image2]

So my generator takes the original inputs and for every epoch resamples these to have a more balanced steering angle distribution as above and then prepares 6 versions of each input for the left/right camera and flipped versions. The training set is much smaller than the original dataset, but the resampling is redone on every epoch in the generator, so eventually the model will see all the examples.

It the end I did not have to add drastic manual steering correction examples for dangerous situations as the left/right camera images and stratified sampling was enough for the network to learn to stay in safe regions.

This data preprocessing worked for both the nvidia network and for the downsampled tiny network as well.

From this I concluded that in real life it would be very important to have a very well balanced training set and not only for the steering angles, but also for different road types, visibility conditions, traffic levels, etc, so that the network would perform well under normal and rare conditions as well.

