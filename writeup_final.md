**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visual1.png "Visualization1"
[image2]: ./examples/visual2.png "Visualization2"
[image3]: ./examples/visual3.png "Visualization3"
[image4]: ./Speed_resize.jpeg "Traffic Sign 1"
[image5]: ./rsz_road_work.jpg "Traffic Sign 2"
[image6]: ./rsz_no_pass.jpg "Traffic Sign 3"
[image7]: ./rsz_image5.jpg "Traffic Sign 4"
[image8]: ./rsz_1image5.jpg "Traffic Sign 5"
[image9]: ./examples/classes.png "Class Types"
[image10]: ./examples/random.png "Random of type"
[image11]: ./examples/norm.png "Before/After preprocessing"
[image12]: ./examples/augment.png "Before/After augmentation"
[image13]: ./rsz_priority_2.jpg "Traffic Sign 6"
[image14]: ./examples/new_images.png "New Traffic Signs"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

***1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.**

You're reading it! and here is a link to my [project code](https://github.com/apoorva-t/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier-Solution.ipynb)

### Data Set Summary & Exploration

**1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.**

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32x32x3)
* The number of unique classes/labels in the data set is 43

**2. Include an exploratory visualization of the dataset.**

Here is an exploratory visualization of the data set. It is a bar histogram showing the count of images of each class type in the training, validation and test sets. It is apparent from this that some classes have fewer samples ~200, while others have 1800+ samples. So our training model may be biased towards the better represented classes.

![alt text][image1]
![alt text][image2]
![alt text][image3]

Also shown are one image each from the 43 classes of signs, along with the sample count of each class.  

![alt text][image9]

Randomly chosen images from a particular class helps in understanding how samples could vary in brightness, contrast, position etc. The training data preprocessing step will use information gained through the visualization step to introduce corrections/perturbations in the data to make the learning model more robust.

![alt text][image10]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

##### Preprocessing

As a first step, I decided to convert the images to grayscale because in the paper pointed to in the assignment, a better accuracy was achieved using a single color channel. Also, I didn't find training with RGB images to give any better results.

The images were normalized as suggested applying the formula (image - 128)/128 to rescale each image in the range [-1,1]. This was found to give better results than image standardization which scales it to a normal distribution with mean=0 and std=1. Training with normalized images showed an improvement in accuracy which proves that feature scaling is an important step in classification problems.

Looking at the images in the visualization step, it is obvious that some images are dark, while some are blurred. So the preprocessing pipeling includes sharpening and constrast improvement steps for these.

Here is an example of an original image, and that after normalization/pre-processing:

![alt text][image11]

##### Augmentation

As pointed out in the Sermanet LeCunn paper, generating additional data by introducing perturbations in the original data helps the training model learn better and detect these distortions in the test data. In the data augmentation step, I am generating a copy of the entire training set and applying random brightness and contrast corrections. In order to make sure that the learning model is robust to deformations in the input data, random rotations between {-20,20} are applied to the augmented set. I am also cropping/padding the 32x32 image to 28x28 so that the learning algorithm can ignore boundary zone of the image which is highly unlikely to be in our region of interest, and focus on the central area - likely to contain the features relevant to the traffic sign. The ranges of perturbation that were found to give the highest validation accuracy were chosen.

Here is an example of an original image and an augmented image:

![alt text][image12]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x32   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16				    |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 3x3x64     |
| RELU					|												|
| Fully connected		| Outputs  120   								|
| RELU					|												|
| Dropout				| keep_prob = 0.5								|
| Fully connected		| Outputs  43   								|

 

I started off by implementing the LeNet architecture that we used in the LeNet lab. In my initial runs, I found that after ~20 epochs of training, my training accuracy shot up to 99.99%, however my validation accuracy was still poor at ~85%. This meant that the model was overfitting. Since dropout and L2 regularization are two of the techinques described in the lessons to prevent overfitting, I applied dropout to the output of the fully connected layers. A keep_prob of 0.5 for the dropout gave best results and led to a good improvement in my validation accuracy. I also added another convolutional layer to the original architecture which gave almost another couple percent improvement in validation accuracy. On increasing the width of the convolutional NN on layers 2 and 3, the accuracy of the model went up. I suspect this is because - as we go deeper in the network the layers try to learn bigger features like objects and an increased width helps in learning more features at this depth.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer - the same as that used in the LeNet lab. A smaller batch size and increasing the number of epochs gave a better accuracy. So after some experimentation with batch sizes between 60-200, a batch size=100 and EPOCHS=30 were chosen to train the model. The weights were initialized from a truncated normal distribution with mean=0 and std=0.1, and biases were initialized zeros.

During the training process, I found that after a few epochs the training and validation accuracy started to oscillate, sometimes going down by almost 1.5-2 percent and then going back up between epochs. This seemed to be because as the learning rate was too high - as it got closer to the minima for the error, a high learning rate seemed to push it beyond the minima. Therefore, I changed the model to lower the learning rate by 1.5 every 7 epochs, starting with an initial learning rate=0.001. This reduced the oscillations in the accuracy considerably (and I suspect was the reason for converging on a better accuracy in fewer epochs).


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ~99.9%
* validation set accuracy of ~96.5% 
* test set accuracy of ~95.8%

I took an iterative approach to finding the solution for this traffic sign classification problem. It began with implementating the LeNet architecture from the lab assignment as a starting point. This was chosen because it is a model already proven to give good results with image classification (with MNIST digits). I first used the normalized RGB images as the input to feed to this model, and then tried using grayscale converted images as the input. I did not find any appreciable difference between training with RGB vs grayscale images, and in fact the performance with grayscale images was slightly better. So after this point, I maintained the grayscale conversion step in my preprocessing pipeline. I suppose this allows the model to focus on traffic sign defining features like shapes, boundaries, lines etc. instead of the color.

On achieving good training accuracy, but poor validation accuracy, the next step I took was to add dropout to the fully connected layer. This helped with the problem of over-fitting of training data and bumped up my validation accuracy. I also tried removing a fully connected layer from the LeNet architecture, so that now my model had only 2 fully connected layers. This led to a slight improvement in the validation accuracy. With a trial and error approach, a batch size of 100 with epochs=20 were used at this point. At this point, I had reached ~93% accuracy on my validation data. 

The next step was to see the effect of augmenting the training data set. I tried this mainly because of the hints given in the stand out suggestions, and because the Sermanet/LeCunn paper showed a decrease in validation error when training with a jittered data set. By introducing rotations, brightness/contrast manipulations, cropping/padding the validation accuracy went further up. 

I was seeing some oscillations in the accuracy between laters epochs, and decided to reduce the learning rate by a factor of 1.5 every 7 epochs, and increase the number of training epochs. A starting rate of 0.001 and epochs=30 were chosen for the final solution. Since I had reached almost 95% validation accuracy, I decided to see how my model worked with the test set - it was giving close to 93%. At this point, I tried a few different approaches to try and break beyond 95% on the validation accuracy - introducing brightness correction in the preprocessing step, playing around with the learning rate, increasing the number of epochs further. None of these gave better results.

I decided to add another convolutional layer based on the architecture described in the Sermanet/LeCunn paper and some posts on the Slack group indicating adding another conv layer increased their accuracy considerably. I was able to go beyond 96% validation accuracy with this step, indicating that deeper conv networks do a better job of learning/generalizing during classification. The final step of increasing the width of the convolutional layers 2 and 3 increased the validation accuracy up to 97%. I suspect this is because the deeper layers were able to learn more features.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are German traffic signs that I found on the web:

![alt text][image14] 

The first image should be fairly easy for the model to classify, and chosen just to prove this as a sanity check. The next image is a road work sign which is blurred and slightly rotated which should make it difficult for the model to classify. The third image is of a no passing traffic sign. This image has bright sunlight shining behind it in the bottom right corner and bears similarities to other circular traffic signs with shapes in the center making it a candidate for mis-classification. The fourth and fifth images are of a warning sign which is taken in dark conditions - one with a text sign below it and one which zooms in on only the warning sign. The sixth image is of the priority road traffic sign. I chose this sign because it is slightly different from the other signs in that it has a diamond shape and color (yellow). It would be interesting to see if this results in mis-classification due to grayscale conversion. Also, this image has traces of other objects - a pole behind it, and small chip of the traffic light.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 30 kmph   | Speed Limit 30 kmph  							| 
| Road work     		| Road work  					  			    |
| No passing		    | No passing									|
| General caution	    | Children crossing  		 			        |
| General caution(crop) | General caution     							|
| Priority road		    | Priority road									|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of ~83%. The image that it misclassfied - general caution - contained some text below the sign. The robustness of the prediction model could be improved by training it with augmented data containing translated versions of the traffic signs, as well as distortions introduced by text or other objects present in the image. The fact that the model was able to predict the rotated road work sign also bolsters the argument for adding more augmented data. The model seems to handle brightness/contrast differences in the images reasonably well.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the last cell of the Ipython notebook. Alongwith the output of the top 5 softmax probabilities in a numpy array, the notebook also contains a visualization of the top 5 predictions in bar charts.

For the first image, the model is sure that this is a 30 kmph speed limit sign (probability of 1.0).

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed Limit 30 kmph  							| 
| .99     				| Road Work   								    |
| .99					| No passing									|
| .53	      			| Children crossing				 				|
| 1.0				    | General Caution      							|
| 1.0				    | Priority Road     							|


For the second image, the model correctly predicted it as a road work sign with a probability of 0.99. The accuracy of prediction on this image can be attributed to introducing rotations in the augmented training data.

For the third image, the model rightly predicted it as a no passing sign with a probability of nearly 0.99, despite a bright spot in the background.

For the fourth image, the model predicted it as a children crossing sign with a probability of 0.53. Its next guess was of a slippery road sign with a probablity of 0.25. The model's third best prediction with a probability of 0.1 was in fact the correct sign - general caution. My guess is that for the model to work better on images like these, we need better training data augmentation by introducing distortions in the form of lateral shifts or presence of other signs/objects in the image.

For the fifth image (which is cropped version of the fourth retaining only the actual sign), the performance was good with a correct prediction of probability = 1.0 for the general caution sign. This is despite the image having a poor contrast.

For the sixth image, the performance was again good with a correct prediction probability of ~1.0 for a priority road sign.

In conclusion, with some more training data augmentation, the model should be able to perform well on real world traffic sign images.


