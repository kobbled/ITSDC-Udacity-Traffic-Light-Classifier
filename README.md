# ITSDC Project : Traffic Light Classifier

In this project, you’ll use your knowledge of computer vision techniques to build a classifier for images of traffic lights! You'll be given a dataset of traffic light images in which one of three lights is illuminated: red, yellow, or green.

| ![all_lights](images/all_lights.png) |
|:---:|
| **Images from the dataset. Left to right: red, green, and yellow traffic lights.** |

#Classification Steps

In the provided notebook, you'll pre-process these images, extract features that will help distinguish the different types of images, and use those features to classify the traffic light images into three categories: red, yellow, or green. The tasks will be broken down into a few sections:

1. Loading and visualizing the data. The first step in any classification task is to be familiar with your data; you'll need to load in the images of traffic lights and visualize them!

2. Pre-processing. The input images and output labels need to be standardized; that is, all the input should be of the same type of data and of the same size, and the output should be a numerical label. This way, you can analyze all the input images using the same procedures, and you know what output to expect when you eventually classify a new image.

| ![processing_steps](images/processing_steps.png) |
|:---:|
| **Pre-processed, standardized images** |

3. Feature extraction. Next, you'll extract some features from each image that will be used to distinguish and classify these images. This is where you have a lot of creativity; features should be 1D vectors or even single values that provide some information about an image that can help classify it as a red, yellow, or green traffic light.

| ![feature_ext_steps](images/feature_ext_steps.png) |
|:---:|
| **An example of feature extraction steps** |

4. Classification and visualizing error. Finally, you'll write one function that uses your features to classify any traffic light image. This function will take in an image and output a label. You'll also be given code to classify a test set of data, compare your predicted label with the true label, and determine the accuracy of your classification model.

5. Evaluate your model. To pass this project, your classifier must be >90% accurate and never classify any red lights as green; it's likely that you'll need to improve the accuracy of your classifier by changing existing features or adding new features. I'd also encourage you to try to get as close to 100% accuracy as possible!

Next, read through the instructions and get ready to build a classifier!
