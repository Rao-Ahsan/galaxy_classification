# galaxy_classification




## Objective
The aim of the project is to develop a program that takes an image as the input and determines
if the provided image is that of a galaxy or not. If it determines that it is a galaxy, it proceeds to
classify which kind of galaxy it is - spiral, elliptical, or irregular.<br/>
## Dependencies
- Python
- 3.9.x 
- TensorFlow 
- Keras 
- NumPy 
- argparse 
- imutils 
- OpenCV 
- Matplotlib 
- scikit-learn 
- random
- os shutil

## Execution
First of all, we needed to set up a neural network before we began any training of sorts. LeNet architecture is the architecture which is most commonly used in binary or categorical classification of images or videos. It is the architecture which we have chosen to develop our neural network. It contains a single static method, with parameters height, width, depth, and number of classes and returns a model at the end. The method is initiated with a sequential model. In the beginning, we have the input layer. Then we add the first convolutional layer, with 20 hidden nodes, followed by a ReLU activation function, followed by a maximum pooling layer. Then we add the second convolutional layer, with 50 hidden nodes, followed similarly by a  ReLU activation function and a maximum pooling layer. Then we add a flatten layer (to convert the 2D images to 1D), followed by a dense layer (to make sure all nodes in the previous layer is connected to each node in the next layer), followed by another ReLU activation function. Finally, we add a softmax activation function in the output layer and create the model.<br/>
Next, we had to write the script to train this network with the images from our dataset. For this, we had initially chosen 25 epochs with a batch size of 32 and an initial learning rate of 0.001 (standard learning rate). We had previously created a folder with four subfolders inside, each containing the images pertaining to that category. The program goes through each image, stores its label in a list, scales the data of the image (between 0 and 1 inclusive), and stores the data in another list. This means each index in the list of labels corresponds to the same index in the list of data. Then it splits the data between training and testing sets. In our program, the testing set was chosen to be 25% of the entire dataset. Our images were also augmented (resized, rotated, blurred, etc.) to train the network to expect minor modifications so that it can classify better and with greater accuracy because in the real world, the images are hardly perfect and will probably have some of these augmentations as well. Finally, it builds the model and proceeds to train it with the training dataset and test it with the testing dataset. At the end, a plot is produced showing the training and value loss and accuracy.<br/>
We had created tens of models before we had found one which showed a very promising result. We had varied the epochs and it was observed that when accuracy was lower, increasing the number of epochs led to a greater accuracy while when accuracy is higher, increasing the epochs does not change the accuracy by much. Hence, after some trial and error, the number of epochs was chosen to be 175. The batch size was also varied but it showed little to no impact and so the default value of 32 was kept. The case was the same with the learning rate. A third set of convolution layers was added. It greatly slowed down the training process, increasing the time taken to train the network by almost double while showing only a little increase in training accuracy. This change was later reverted. The training-testing split was increased to 50% and this showed a decrease in accuracy so it was not kept. When the network was trained with all these modifications, the following plot was produced<br/>

![image](https://user-images.githubusercontent.com/65136938/154564927-10555a31-2fdc-49a6-b0c6-6758123b6b70.png).<br/>
As can be observed, the training loss decreased significantly while the training accuracy increased by an almost equal percentage. The value loss increased with time but this is expected because the more images are fed to the network, the greater the value loss will be. The value accuracy, on the other hand, remained almost constant throughout the process.<br/>

The pre-trained model provided in the download folder has an accuracy of 94.4%. It was trained with 20,000 images from the Kaggle dataset. A snapshot of the terminal at the end of the training process is given below to show this result.<br/>
![image](https://user-images.githubusercontent.com/65136938/154565077-89595c13-10bc-4ec2-a595-0407adac85ba.png).<br/>
Afterwards came the testing process. Initially, we had thought of only testing one single image instead of classifying and sorting a huge dataset. Hence, our first test script was written to take an input image, resize it to 28x28 pixels, and feed it to the model so that the model could predict its label. Then, depending on which label gets the highest probability, the image is assigned the label. This, along with the probability of the image belonging to the assigned label, is displayed on the output image. An example of an output image is displayed below.<br/>
![image](https://user-images.githubusercontent.com/65136938/154565135-0bdea0a2-eb8f-4f93-a92c-ed0b53a8a01a.png).<br/>
Then the idea occurred to us to extend the program so that a person could classify multiple images without having to pass every single image through the program above and manually copying it into the desired folder. The script was written in such a way that five subfolders were already created in a particular folder. One of the five subfolders contained all the images to be classified and sorted. The other subfolders were empty at the beginning. The script goes through each image in the subfolder and feeds it into the model. The model predicts the label of that image and assigns a path to it to where it should be copied. This way the model can easily classify over 20 images per second without having the need to display each and every one with its probability.<br/>

## Uses:
This project can be used to classify a singular image. For example, if someone wants to see whether a particular image is that of a galaxy or not, this program could be used to identify which galaxy it most closely resembles to and the probability of likeness of the picture to that galaxy. A second usage would be to classify a huge dataset of images. For example, let us consider that a person has 100,000 images of different galaxies on his hard drive and he/she wants to sort these images into respective folders depending on the category of the galaxy. It would be a very time-consuming process for that person to identify each image individually and then sort them. This program could be used to look through each image, calculate its probability of resemblance to a galaxy, and sort it into the correct folder. This could save researchers, scientists, astronomers, etc. hundreds of hours of perusal and classification.<br/>












