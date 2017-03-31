# Behavioral Cloning
---

## Goal
---
The goals of this project are

1. Use the simulator to collect data
2. Build a model in Keras that predicts steering angles from camera images
3. Train and validate the model with a training and validation set
4. Test the model on track one (the car should not leave the road)
5. Summarize the results in a written report.

## Contents of this submission
---
1. this README.md
2. model.ipynb: source code with short explanations of the development process
4. model.json
5. model.h5
6. network.py: contains the code for creating the net
7. Database.py: contains the code for accessing the CSV files, image processing, imitating different camera positions, and providing the generators for the training of the net.
8. drive.py


## Network Architecture
---
The network that was used for the submission is derived from the proposal by Mariusz Bojarski et al, arXiv:1604.07316v1 [cs.CV] 25 Apr 2016. Input layer is a lambda layer that takes input images at a resolution of 135x320x3 and normalizes values into a range -0.5...0.5. The reduction of the resolution as compared to the original image resolution of 160x320x3 was performed in order to get rid of the "selfie" of the car in the lower part of the image, which would have been located in different positions in the image for different camera positions (left/center/right). The main part of the network consists of 5 convolutional and 4 dense layers, which are connected by maxpooling layers. For activation, I use relu layers and finally tanh layers. The output of the network is a dense layer with one output channel corresponsind to the value of the steering angle, normalized to the range -1...1 (tanh activation layer). In order to prevent the output layer from being forced to train output values +-1.0, corresponding to infinite input values, I rescale all normalized steering angles to the range -0.9 ... 0.9 for the training. For driving the car, I perform an inverse rescaling in drive.py.

In order to prevent overfitting, I use several dropout layers.

The network is created by functions defined in network.py, where there are several versions that I tried during development. Relevant for the discussion here is only the network created by the function **`createModel6(lr)`**.

In Detail:

1. Input layer is a lambda layer for normalization to values -0.5...0.5
2. 5x5 convolution with 24 output channels
3. relu
4. 2x2 maxpooling
5. dropout
6. 5x5 conv, 36 output
7. relu
8. 2x2 maxpooling
9. dropout
10. 5x5 conv, 48 output
11. relu
12. 2x2 maxpool
13. drouput
14. 3x3 conv, 64 output
15. relu
16. dropout
17. flatten
18. dense 200
19. relu
20. dropout
21. dense 50
22. tanh
23. dropout
24. dense 10
25. tanh
26. dropout
27. dense 1
28. tanh: output layer

## Data for Learning and Training the Network
---
For creating the data set for training the network there were two versions of the simulator available: the "full" version with center/left/right camera, and a beta version with only the center camera. _Unfortunately, the "full" version did not run on my computer_: it hang during startup. So I was left with the beta version. 

Since the simulator is able to record a sequence of images together with the steering angle (besides other information), it is possible to train a network based on this information. As described in the lecture as well as in the Bojarski paper, it is necessary to teach the network to recover from error, which means to re-center the car on the street when it got off-center. At this point, the off-center cameras come into play: An image from the right (left) camera is off-center, therefor one can add negative (positive) offset to the steering angle during training. 

Now, these off-center cameras where not available for data recording. Therefore I took a different approach: I imitated several cameras in different positions by taking sub-images from the center camera, keeping the horizon in a fixed relative position, and rescaling the sub-images to the original size (see function **`subImg(img, scale, center, hpos)`** in **`Database.py`**). With that approach I imitate 10 additional cameras (5 to the left and 5 to the right of the center camera). Assuming a steering angle normalized to the range -1...1, I add a correction angle for each imitated side camera with values that can be found in function **`gen_y_beta(db, delta, bias, rescale)`** in **`Database.py`**.

The data set that I finally used to train the network consists of two image sequences, one driving the car in the direction it was initially standing on the road, one in opposite direction, plus the image series provided by Udacity. Together with the imitated cameras I use 381498 images in total, which I split in 80% training data and 20% validation data. For training the network I use the adam optimizer with reduced learning rate and a batch size of 80 images. Furthermore, I flip 50% of the images (compare **`genData()`** in **`Database.py`**) to get an equal distribution of negative and positive steering angles. ** Examples for the images can be found in the jupyter notebook model.ipynb. I understand that images in this report would improve the document and help understand the text, but I nevertheless would like to skip this step because it is somewhat redundant -- figures already contained in the notebook -- and time consuming. As I already repeatedly discussed with my mentor, I have only very limited time for this course.**

The reduction of the learning rate was chosen by trial and error; although this should not be necessary, I have the feeling that this led to a better convergence of the learning process. The mse of the validation set always stays above the mse of the training set, which I take as an indication for overfitting; best results here again with lr=0.00025.

The model for the hand-in was trained with lr=0.00025 and 8 epochs. mse for training data was 0.0071 and for validation data 0.0096 after the final epoch. I fixed this result after a number of experiments with lr and number of epochs, starting with lr=0.001 down to 0.0001, and number of epochs between 4 and 16. In all other cases I had higher values for the validation mse, as well as a ratio of validation mse / training mse >= 2, which implies stronger overfitting, to my understanding. What I handed in was the best I could acheive, in terms of behavior of the car on the road as well as in terms of validation mse as well as in terms of ratio validation / training mse. ** The reviewer pointed out that the claimed 8 epochs were in disagreement with the 5 epochs that were used in the code. It is true that I did further experiments but could not improve on the validation error that I had acheived before, and I left the number unchanged in the code. It is now 8 epochs in the code, this point is corrected. For future developments I will consider the function ModelCheckpoint as suggested by the reviewer, this will of course be very useful. **

## Simulation
---
The car runs nicely in the direction it initially stands on the road, it does not leave the track and is able to recover from error. I induced error by manually driving the car off-center. The car was able to even recover from situations where some wheels already were off the street. It also stays on the road when driving in the opposite direction.




