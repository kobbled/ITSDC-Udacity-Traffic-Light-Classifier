
# coding: utf-8

# # Traffic Light Classifier
# ---
# 
# In this project, you’ll use your knowledge of computer vision techniques to build a classifier for images of traffic lights! You'll be given a dataset of traffic light images in which one of three lights is illuminated: red, yellow, or green.
# 
# In this notebook, you'll pre-process these images, extract features that will help us distinguish the different types of images, and use those features to classify the traffic light images into three classes: red, yellow, or green. The tasks will be broken down into a few sections:
# 
# 1. **Loading and visualizing the data**. 
#       The first step in any classification task is to be familiar with your data; you'll need to load in the images of traffic lights and visualize them!
# 
# 2. **Pre-processing**. 
#     The input images and output labels need to be standardized. This way, you can analyze all the input images using the same classification pipeline, and you know what output to expect when you eventually classify a *new* image.
#     
# 3. **Feature extraction**. 
#     Next, you'll extract some features from each image that will help distinguish and eventually classify these images.
#    
# 4. **Classification and visualizing error**. 
#     Finally, you'll write one function that uses your features to classify *any* traffic light image. This function will take in an image and output a label. You'll also be given code to determine the accuracy of your classification model.    
#     
# 5. **Evaluate your model**.
#     To pass this project, your classifier must be >90% accurate and never classify any red lights as green; it's likely that you'll need to improve the accuracy of your classifier by changing existing features or adding new features. I'd also encourage you to try to get as close to 100% accuracy as possible!
#     
# Here are some sample images from the dataset (from left to right: red, green, and yellow traffic lights):
# <img src="images/all_lights.png" width="50%" height="50%">
# 

# ---
# ### *Here's what you need to know to complete the project:*
# 
# Some template code has already been provided for you, but you'll need to implement additional code steps to successfully complete this project. Any code that is required to pass this project is marked with **'(IMPLEMENTATION)'** in the header. There are also a couple of questions about your thoughts as you work through this project, which are marked with **'(QUESTION)'** in the header. Make sure to answer all questions and to check your work against the [project rubric](https://review.udacity.com/#!/rubrics/1213/view) to make sure you complete the necessary classification steps!
# 
# Your project submission will be evaluated based on the code implementations you provide, and on two main classification criteria.
# Your complete traffic light classifier should have:
# 1. **Greater than 90% accuracy**
# 2. ***Never* classify red lights as green**
# 

# # 1. Loading and Visualizing the Traffic Light Dataset
# 
# This traffic light dataset consists of 1484 number of color images in 3 categories - red, yellow, and green. As with most human-sourced data, the data is not evenly distributed among the types. There are:
# * 904 red traffic light images
# * 536 green traffic light images
# * 44 yellow traffic light images
# 
# *Note: All images come from this [MIT self-driving car course](https://selfdrivingcars.mit.edu/) and are licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).*

# ### Import resources
# 
# Before you get started on the project code, import the libraries and resources that you'll need.

# In[2]:


import helpers # helper functions

import random
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images
import cv2 # computer vision library


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Training and Testing Data
# 
# All 1484 of the traffic light images are separated into training and testing datasets. 
# 
# * 80% of these images are training images, for you to use as you create a classifier.
# * 20% are test images, which will be used to test the accuracy of your classifier.
# * All images are pictures of 3-light traffic lights with one light illuminated.
# 
# ## Define the image directories
# 
# First, we set some variables to keep track of some where our images are stored:
# 
#     IMAGE_DIR_TRAINING: the directory where our training image data is stored
#     IMAGE_DIR_TEST: the directory where our test image data is stored

# In[3]:


# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"


# ## Load the datasets
# 
# These first few lines of code will load the training traffic light images and store all of them in a variable, `IMAGE_LIST`. This list contains the images and their associated label ("red", "yellow", "green"). 
# 
# You are encouraged to take a look at the `load_dataset` function in the helpers.py file. This will give you a good idea about how lots of image files can be read in from a directory using the [glob library](https://pymotw.com/2/glob/). The `load_dataset` function takes in the name of an image directory and returns a list of images and their associated labels. 
# 
# For example, the first image-label pair in `IMAGE_LIST` can be accessed by index: 
# ``` IMAGE_LIST[0][:]```.
# 

# In[4]:


# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)


# ## Visualize the Data
# 
# The first steps in analyzing any dataset are to 1. load the data and 2. look at the data. Seeing what it looks like will give you an idea of what to look for in the images, what kind of noise or inconsistencies you have to deal with, and so on. This will help you understand the image dataset, and **understanding a dataset is part of making predictions about the data**.

# ---
# ### Visualize the input images
# 
# Visualize and explore the image data! Write code to display an image in `IMAGE_LIST`:
# * Display the image
# * Print out the shape of the image 
# * Print out its corresponding label
# 
# See if you can display at least one of each type of traffic light image – red, green, and yellow — and look at their similarities and differences.

# In[5]:


## TODO: Write code to display an image in IMAGE_LIST (try finding a yellow traffic light!)
## TODO: Print out 1. The shape of the image and 2. The image's label

def getImage(index=0, color=None, Random=False, showImg=False):
    #get total number of images in set
    total_num_of_images = len(IMAGE_LIST)
    # check whether color is red green of yellow
    acceptable_colors = tuple(['red', 'yellow', 'green'])
    if (not color in acceptable_colors) or color == None:
            raise ValueError('color {} is not an acceptable color'.format(color))
    # pick random index if random, and keep choosign random image until correct color is found
    if Random == True:
        image_index = random.randint(0, total_num_of_images-1)
        if (color != None):
            while IMAGE_LIST[image_index][1] != color:
                image_index = random.randint(0, total_num_of_images-1)
    # incement index until color is found
    else:
        image_index = index
        while IMAGE_LIST[image_index][1] != color:
            if image_index >= total_num_of_images:
                total_num_of_images = 0
            else:
                total_num_of_images += 1
    
    # select image
    selected_image = IMAGE_LIST[image_index][0]
    # select label
    selected_label = IMAGE_LIST[image_index][1]
    
    if showImg:
        plt.imshow(selected_image)
        print("Shape: {0[1]}px X {0[0]}px".format(selected_image.shape))
        print("Label: " + str(selected_label))
        print("Index: " + str(image_index))
    return selected_image, selected_label, image_index 


image, label, index = getImage(color='red', Random=True, showImg=True)
image2, label2, index3 = getImage(color='yellow', Random=True, showImg=True)
image3, label2, index3 = getImage(color='green', Random=True, showImg=True)


# # 2. Pre-process the Data
# 
# After loading in each image, you have to standardize the input and output!
# 
# ### Input
# 
# This means that every input image should be in the same format, of the same size, and so on. We'll be creating features by performing the same analysis on every picture, and for a classification task like this, it's important that **similar images create similar features**! 
# 
# ### Output
# 
# We also need the output to be a label that is easy to read and easy to compare with other labels. It is good practice to convert categorical data like "red" and "green" to numerical data.
# 
# A very common classification output is a 1D list that is the length of the number of classes - three in the case of red, yellow, and green lights - with the values 0 or 1 indicating which class a certain image is. For example, since we have three classes (red, yellow, and green), we can make a list with the order: [red value, yellow value, green value]. In general, order does not matter, we choose the order [red value, yellow value, green value] in this case to reflect the position of each light in descending vertical order.
# 
# A red light should have the  label: [1, 0, 0]. Yellow should be: [0, 1, 0]. Green should be: [0, 0, 1]. These labels are called **one-hot encoded labels**.
# 
# *(Note: one-hot encoding will be especially important when you work with [machine learning algorithms](https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/)).*
# 
# <img src="images/processing_steps.png" width="80%" height="80%">
# 

# ---
# <a id='task2'></a>
# ### (IMPLEMENTATION): Standardize the input images
# 
# * Resize each image to the desired input size: 32x32px.
# * (Optional) You may choose to crop, shift, or rotate the images in this step as well.
# 
# It's very common to have square input sizes that can be rotated (and remain the same size), and analyzed in smaller, square patches. It's also important to make all your images the same size so that they can be sent through the same pipeline of classification steps!

# In[6]:


# calculate the average of a lower bounded array of numbers
# for classifying traffic light images this lower bound was 
# set to 0.1 radians.
def average(array, lBound=0.1, uBound=0.6):
    new_array = []
    for i,row in enumerate(array):
        for j,cell in enumerate(row):
            if (cell >= lBound and cell < uBound):
                new_array.append(cell)
    if len(new_array) == 0:
        return 0
    else:
        return sum(new_array)/len(new_array)
    
def to_polar_coords(xpix, ypix):
    # Calculate distance to each pixel
    dist = np.sqrt(xpix**2 + ypix**2)
    # Calculate angle using arctangent function
    angles = np.arctan2(ypix, xpix)
    return dist, angles

# Use a sobel image filter and return the magnitude and angles vectors
# for the image
# reference : https://www.programcreek.com/python/example/89325/cv2.Sobel
def sobel_filter(image):
    # sobel filter x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    # absolute the gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    
    # get magnitude array & angle array
    mag, ang = to_polar_coords(abs_sobelx, abs_sobely)
    
     # scale values to 8 bit
    scale_factor = np.max(mag)/255
    mag = (mag/scale_factor).astype(np.uint8)
    
    return mag, ang

def getBoundingBox(img_thres):
    # ref : https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(img_thres > 0))
    rect = cv2.minAreaRect(coords)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    angle = rect[-1]
    # handle the angle to get correct output
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    return box, angle
            
    
def rectanglemask(img, box):
    # transpose as box is flipped
    mask_img = cv2.transpose(np.zeros_like(img))
    cv2.drawContours(mask_img,[box], 0, (255,255,255), -1)
    mask_img = np.uint8(mask_img)
    ret, mask = cv2.threshold(mask_img,1,255,cv2.THRESH_BINARY)
    # transpose back
    mask = cv2.transpose(mask)
    return mask

# rotate the image by a specified angle.
# ref: https://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/
def rotate(image, angle):
    if len(image.shape) > 2:
        h, w, _ = image.shape
    else:
        h, w = image.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    return rotated
    
    
# Some of the traffic light images are very skewed. This function attempts to
# center the traffic lights for easier feature extraction. 
# This function filters and image through a sobel image filter extracting magnitude vectors from
# each pixel. These vectors are then filtered and an average image rotation angle is computed
# and returned.
# The magnitude thresold, and angle threshold values were randomly choosen though trail and error.
def getTrafficLightAngle(image, mag_thes=(80, 255), ang_thres=(-np.pi/6, np.pi/6)):
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    mag, ang = sobel_filter(gray)
        ### print(average(mag, mag_thes[0], mag_thes[1]))
    
     # threshold between angle values, and magnitude values
    ang_threshold = np.zeros_like(ang)
    above_ang =   (ang > ang_thres[0])                 & (ang <= ang_thres[1])                 & (mag > mag_thes[0])                   & (mag <= mag_thes[1])
    ang_threshold[above_ang] = 1
    
    # mask the thresheld image with the angle image
    masked_ang = np.copy(ang)
    masked_ang[ang_threshold == 0] = 0
    
    h, w = ang_threshold.shape
    # If the thresholding only revelas a few pixels disregard
    # and set angle to zero
    numOfEdges = np.count_nonzero(masked_ang>0)
    if numOfEdges > 80:
        box, angle = getBoundingBox(ang_threshold)
    else:
        angle = 0
        box = np.array([[0, w],
                        [0, 0],
                        [h, 0],
                        [h, w]], np.int32)
    
    return box, angle, masked_ang
  
# CODE TEST
# --------------
red, _, index = getImage(color='green', Random=True)
standard_im = np.copy(red)
box, angle, sobel = getTrafficLightAngle(standard_im)
# plt.imshow(sobel)

# create rectangluar mask from edge detection
mask = rectanglemask(sobel, box)
mask = cv2.resize(mask, (red.shape[1], red.shape[0])) 
mask_img = np.copy(red)
#mask image
mask_img = cv2.bitwise_and(red, mask_img,mask = mask)

# rotate image by skew
cnt = rotate(mask, angle)
# get bounding box
x,y,w,h = cv2.boundingRect(cnt)
#rotate masked image & crop
rotated = rotate(mask_img, angle)
cropped = rotated[y:y+h,x:x+w]

print("angle: {}".format(angle))
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
ax1.set_title('image')
ax1.imshow(red)
ax2.set_title('mask')
ax2.imshow(mask_img)
ax3.set_title('cropped')
ax3.imshow(cropped)
plt.show()


# In[8]:


# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):
    ## TODO: Resize image and pre-process so that all "standard" images are the same size  
    box, angle, sobel = getTrafficLightAngle(image)
    
    # create rectangluar mask from edge detection
    mask = rectanglemask(sobel, box)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0])) 
    mask_img = np.copy(image)
    #mask image
    mask_img = cv2.bitwise_and(image, mask_img,mask = mask)

    # rotate image by skew
    cnt = rotate(mask, angle)
    # get bounding box
    x,y,w,h = cv2.boundingRect(cnt)
    #rotate masked image & crop
    rotated = rotate(mask_img, angle)
    cropped = rotated[y:y+h,x:x+w]
    
    #resize to 32 X 32
    standard_im = cv2.resize(cropped, (32, 32))
    
    return standard_im
    

#original red green and blue image
random_red, _, index1 = getImage(color='red', Random=True)
random_yellow, _, index2 = getImage(color='yellow', Random=True)
random_green, _, index3 = getImage(color='green', Random=True)

# show red, yellow, green images
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
ax1.set_title('red')
ax1.imshow(random_red)
ax2.set_title('yellow')
ax2.imshow(random_yellow)
ax3.set_title('green')
ax3.imshow(random_green)
plt.show()
    
#original red green and blue image
std_red = standardize_input(random_red)
std_yellow = standardize_input(random_yellow)
std_green = standardize_input(random_green)

# show red, yellow, green images
f2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
ax1.set_title('red')
ax1.imshow(std_red)
ax2.set_title('yellow')
ax2.imshow(std_yellow)
ax3.set_title('green')
ax3.imshow(std_green)
plt.show()


# ## Standardize the output
# 
# With each loaded image, we also specify the expected output. For this, we use **one-hot encoding**.
# 
# * One-hot encode the labels. To do this, create an array of zeros representing each class of traffic light (red, yellow, green), and set the index of the expected class number to 1. 
# 
# Since we have three classes (red, yellow, and green), we have imposed an order of: [red value, yellow value, green value]. To one-hot encode, say, a yellow light, we would first initialize an array to [0, 0, 0] and change the middle value (the yellow value) to 1: [0, 1, 0].
# 

# ---
# <a id='task3'></a>
# ### (IMPLEMENTATION): Implement one-hot encoding

# In[9]:


## TODO: One hot encode an image label
## Given a label - "red", "green", or "yellow" - return a one-hot encoded label

# Examples: 
# one_hot_encode("red") should return: [1, 0, 0]
# one_hot_encode("yellow") should return: [0, 1, 0]
# one_hot_encode("green") should return: [0, 0, 1]

def one_hot_encode(label):
    
    ## TODO: Create a one-hot encoded label that works for all classes of traffic lights
    one_hot_encoded = [0, 0, 0]
    # check whether color is red green of yellow
    acceptable_colors = tuple(['red', 'yellow', 'green'])
    if (not label in acceptable_colors):
            raise ValueError('label: {} is not an acceptable color'.format(label))
    if label == 'red':
        one_hot_encoded[0] = 1
    if label == 'yellow':
        one_hot_encoded[1] = 1
    if label == 'green':
        one_hot_encoded[2] = 1
    
    return one_hot_encoded


# ### Testing as you Code
# 
# After programming a function like this, it's a good idea to test it, and see if it produces the expected output. **In general, it's good practice to test code in small, functional pieces, after you write it**. This way, you can make sure that your code is correct as you continue to build a classifier, and you can identify any errors early on so that they don't compound.
# 
# All test code can be found in the file `test_functions.py`. You are encouraged to look through that code and add your own testing code if you find it useful!
# 
# One test function you'll find is: `test_one_hot(self, one_hot_function)` which takes in one argument, a one_hot_encode function, and tests its functionality. If your one_hot_label code does not work as expected, this test will print ot an error message that will tell you a bit about why your code failed. Once your code works, this should print out TEST PASSED.

# In[10]:


# Importing the tests
import test_functions
tests = test_functions.Tests()

# Test for one_hot_encode function
tests.test_one_hot(one_hot_encode)


# ## Construct a `STANDARDIZED_LIST` of input images and output labels.
# 
# This function takes in a list of image-label pairs and outputs a **standardized** list of resized images and one-hot encoded labels.
# 
# This uses the functions you defined above to standardize the input and output, so those functions must be complete for this standardization to work!
# 

# In[11]:


def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
        
    return standard_list

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)


# ## Visualize the standardized data
# 
# Display a standardized image from STANDARDIZED_LIST and compare it with a non-standardized image from IMAGE_LIST. Note that their sizes and appearance are different!

# In[12]:


## TODO: Display a standardized image and its label
total_num_of_images = len(STANDARDIZED_LIST)
image_index = random.randint(0, total_num_of_images-1)

f2, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title(STANDARDIZED_LIST[image_index][1])
ax1.imshow(STANDARDIZED_LIST[image_index][0])
ax2.set_title(IMAGE_LIST[image_index][1])
ax2.imshow(IMAGE_LIST[image_index][0])


# # 3. Feature Extraction
# 
# You'll be using what you now about color spaces, shape analysis, and feature construction to create features that help distinguish and classify the three types of traffic light images.
# 
# You'll be tasked with creating **one feature** at a minimum (with the option to create more). The required feature is **a brightness feature using HSV color space**:
# 
# 1. A brightness feature.
#     - Using HSV color space, create a feature that helps you identify the 3 different classes of traffic light.
#     - You'll be asked some questions about what methods you tried to locate this traffic light, so, as you progress through this notebook, always be thinking about your approach: what works and what doesn't?
# 
# 2. (Optional): Create more features! 
# 
# Any more features that you create are up to you and should improve the accuracy of your traffic light classification algorithm! One thing to note is that, to pass this project you must **never classify a red light as a green light** because this creates a serious safety risk for a self-driving car. To avoid this misclassification, you might consider adding another feature that specifically distinguishes between red and green lights.
# 
# These features will be combined near the end of his notebook to form a complete classification algorithm.

# ## Creating a brightness feature 
# 
# There are a number of ways to create a brightness feature that will help you characterize images of traffic lights, and it will be up to you to decide on the best procedure to complete this step. You should visualize and test your code as you go.
# 
# Pictured below is a sample pipeline for creating a brightness feature (from left to right: standardized image, HSV color-masked image, cropped image, brightness feature):
# 
# <img src="images/feature_ext_steps.png" width="70%" height="70%">
# 

# ## RGB to HSV conversion
# 
# Below, a test image is converted from RGB to HSV colorspace and each component is displayed in an image.

# In[13]:


# Convert and image to HSV colorspace
# Visualize the individual color channels

image_num = random.randint(0, total_num_of_images-1)
test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]

# Convert to HSV
hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)

# Print image label
print('Label [red, yellow, green]: ' + str(test_label))

# HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

# Plot the original image and the three channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('Standardized image')
ax1.imshow(test_im)
ax2.set_title('H channel')
ax2.imshow(h, cmap='gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap='gray')
ax4.set_title('V channel')
ax4.imshow(v, cmap='gray')

# rgb channels
r = test_im[:,:,0]
g = test_im[:,:,1]
b = test_im[:,:,2]

# Plot the original image and the three channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('Standardized image')
ax1.imshow(test_im)
ax2.set_title('R channel')
ax2.imshow(r, cmap='gray')
ax3.set_title('G channel')
ax3.imshow(g, cmap='gray')
ax4.set_title('B channel')
ax4.imshow(b, cmap='gray')


# ---
# <a id='task7'></a>
# ### (IMPLEMENTATION): Create a brightness feature that uses HSV color space
# 
# Write a function that takes in an RGB image and returns a 1D feature vector and/or single value that will help classify an image of a traffic light. The only requirement is that this function should apply an HSV colorspace transformation, the rest is up to you. 
# 
# From this feature, you should be able to estimate an image's label and classify it as either a red, green, or yellow traffic light. You may also define helper functions if they simplify your code.

# In[14]:


### Helper functions
#### List multiplication

def matrix_scalar_mul(matrix, scalar):
    new = []
    for i in range(len(matrix)):
        new.append(matrix[i] * scalar)
    return new


def matrix_multiplication(matrixA, matrixB):
    product = []
    
    if len(matrixA) != len(matrixB):
        raise ValueError('list must be the same size, A:', len(matrixA), 'B:', len(matrixB))

    for i in range(len(matrixA)):
        product.append(matrixA[i] * matrixB[i])

    return product

########################


# return list index which has a maximum value
def max_idx(yvals, ranges):
    mx = 0
    j = 0
    for i in ranges:
        if yvals[i] > mx:
            mx = yvals[i]
            j = i
    return j

# return list index which has a minimum value
def min_idx(yvals, ranges):
    mn = max(yvals)
    j = 0
    for i in ranges:
        if yvals[i] < mn and yvals[i] > 0:
            mn = yvals[i]
            j = i
    return j

# return the top x indicies from a list 
def max_idx_rank(yvals):
    indicies = set(range(len(yvals)))
    # creat a list to append max bins
    max_list = []
    # create set to perform set operations
    max_set = set()
    intersect = indicies - max_set
    
    # rank first 8 bins
    for i in range(0, 8):
        # append next maximum value
        max_list.append(max_idx(yvals, intersect))
        # add value to set list
        max_set.add(max_list[-1])
        # remove bin from bin list
        intersect = indicies - max_set
    
    return max_list

# function to determine if the distribution is bimodal or normal
def is_bimodal(max_list, values):
    difference = []
    for i in range(len(max_list)-1):
        for j in range(max_list[i], max_list[i+1]):
            if values[j] == 0:
                return True
    
    return False

# # find the mean index
# def mean_bin(values):
#     #change zeros to nan for mean
#     mean_vals = [np.nan if x == 0 else x for x in values]
#     mean = np.nanmean(mean_vals)
#     print(mean)
#     difference = []
#     for i in range(len(values)):
#         difference.append(abs(values[i] - mean))
        
#     mean_i = min_idx(difference, range(len(difference)))
    
#     return mean_i
    
    
    


# In[15]:


## TODO: Create a brightness feature that takes in an RGB image and outputs a feature vector and/or value
## This feature should use HSV colorspace values


def color_channels(image, channel, n_bins = 24):
    if channel == "hsv":
        # Convert absimage to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Create color channel histograms
        h_hist = np.histogram(hsv[:,:,0], bins=n_bins, range=(0, 180))
        s_hist = np.histogram(hsv[:,:,1], bins=n_bins, range=(0, 256))
        v_hist = np.histogram(hsv[:,:,2], bins=n_bins, range=(0, 256))
        
        return h_hist, s_hist, v_hist 
    else:
        rgb = np.copy(image)
        # Create color channel histograms
        r_hist = np.histogram(rgb[:,:,0], bins=n_bins, range=(0, 256))
        g_hist = np.histogram(rgb[:,:,1], bins=n_bins, range=(0, 256))
        b_hist = np.histogram(rgb[:,:,2], bins=n_bins, range=(0, 256))
        
        return r_hist, g_hist, b_hist
    
def color_isolate(image, channel):
    if channel == "hsv":
        # Convert absimage to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Create color channels
        h = hsv[:,:,0]
        s = hsv[:,:,1]
        v = hsv[:,:,2]
        
        return h, s, v 
    else:
        rgb = np.copy(image)
        # Create color channels
        r = rgb[:,:,0]
        g = rgb[:,:,1]
        b = rgb[:,:,2]
        
        return r, g, b 

# plot histograms of rbg, or hsv space along the y-axis of the image
def yaxis_hists(rgb_image, channel):
    # seperate image out into different channels of color space
    c1, c2, c3 = color_isolate(rgb_image, 'hsv')
    # Sum components over all coloumns for each row (axis = 1)
    hist_sum = []
    c1_sum = np.sum(c1[:,:], axis=1)
    c2_sum = np.sum(c2[:,:], axis=1)
    c3_sum = np.sum(c3[:,:], axis=1)
    
    #get baselines
    base1 = np.median(c1_sum)
    base2 = np.median(c2_sum)
    base3 = np.median(c3_sum)
    
    # split histrogram around the median
    c1_norm = matrix_scalar_mul((c1_sum - base1).tolist(), -1)
    c2_norm = (c2_sum - base2).tolist()
    c3_norm = (c3_sum - base3).tolist()
    
    # get rid of negative values
    #np.nan
    c1_norm = [0 if x < 0 else x for x in c1_norm]
    c2_norm = [0 if x < 0 else x for x in c2_norm]
    c3_norm = [0 if x < 0 else x for x in c3_norm]
    
    # package as 2D list
    hist_vals = []
    hist_vals.append(c1_norm)
    hist_vals.append(c2_norm)
    hist_vals.append(c3_norm)
    
    # get bins
    bin_edges = range(rgb_image.shape[0])

    return bin_edges, hist_vals

def value_hists(rgb_image, channel):
    c1, c2, c3 = color_channels(rgb_image, channel)
    
    hist_vals = []
    hist_vals.append(c1[0])
    hist_vals.append(c2[0])
    hist_vals.append(c3[0])
    
    # Generating bin centers
    bin_edges = c1[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    
    return bin_centers, hist_vals
    

def plotHist(bins, values, thickness, channel, yLbl, yLim = False):
    if channel == 'hsv':
        names = ('hue','saturation','value')
    else:
        names = ('red','green','blue')
        
    plt.rcdefaults()
    f1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
    
    #first channel
    hist1 = ax1.barh(bins, values[0], align="center", height=thickness)
    # get mean and bimodal bool
    max_list = max_idx_rank(values[0])
    mean = max_list[0]
    bimodal = is_bimodal(max_list, values[0])
    
    ax1.invert_yaxis()
    ax1.set_ylabel(yLbl)
    ax1.set_xlabel("amount")
    ax1.set_title("%s"%names[0])
    ax1.set_title("%s :mu = %i, bi = %r" %(names[0],mean, bimodal))
    if yLim:
        ax1.set_xlim(0, 256)
            
    #second channel
    hist2 = ax2.barh(bins, values[2], align="center", height=thickness)
    # get mean and bimodal bool
    max_list2 = max_idx_rank(values[2])
    mean2 = max_list2[0]
    bimodal2 = is_bimodal(max_list2, values[2])
    
    ax2.invert_yaxis()
    ax2.set_ylabel(yLbl)
    ax2.set_xlabel("amount")
    ax2.set_title("%s :mu = %i, bi = %r" %(names[2],mean2, bimodal2))
    if yLim:
        ax2.set_xlim(0, 256)
        
    #third channel
    sat_val = matrix_multiplication(values[2], values[0])
    hist3 = ax3.barh(bins, sat_val, align="center", height=thickness)
    # get mean and bimodal bool
    max_list3 = max_idx_rank(sat_val)
    mean3 = max_list3[0]
    bimodal3 = is_bimodal(max_list3, sat_val)
    
    ax3.invert_yaxis()
    ax3.set_ylabel(yLbl)
    ax3.set_xlabel("amount")
    ax3.set_title("%s :mu = %i, bi = %r" %('hue+val',mean3, bimodal3))
    if yLim:
        ax3.set_xlim(0, 256)
    
    plt.show()


# In[16]:


def feature_value(rgb_image, plot = False):
    ## TODO: Convert absimage to HSV color space
    ## TODO: Create and return a feature value and/or vector
    
    # calculate HSVspace over the height on the traffic light
    bins, values = yaxis_hists(rgb_image, 'hsv')
    
    # for testing purposes
    if plot == True:
        plotHist(bins, values, 0.8, 'hsv', 'y-dist')
    
    return values[2]

# image, label, image_num = getImage(color='yellow', Random=True, showImg=False)
# #image_num = random.randint(0, total_num_of_images-1)
# test_im = STANDARDIZED_LIST[image_num][0]
# test_label = STANDARDIZED_LIST[image_num][1]

feature = feature_value(test_im, True)
print(test_label)
plt.imshow(test_im)


# ## (Optional) Create more features to help accurately label the traffic light images

# In[17]:


def feature_valueXHue(rgb_image, plot = False):
    ## TODO: Convert absimage to HSV color space
    ## TODO: Create and return a feature value and/or vector
    
    # calculate HSVspace over the height on the traffic light
    bins, values = yaxis_hists(rgb_image, 'hsv')
    hue_val = matrix_multiplication(values[2], values[0])
    
    return hue_val


# In[18]:


# (Optional) Add more image analysis and create more features
#####
# This feature will look at the hue of the image. If the max hue is within a 
# color range it will return that color.
#####
# hue ranges
    # * red = (0, 10) || (160, 180)
    # * yellow = (90, 130) 
    # * green = (80, 140) 
    
def feature_rgb(rgb_image, plot = False):
    ## TODO: Convert absimage to HSV color space
    ## TODO: Create and return a feature value and/or vector
    
#     # calculate hsvspace hue over the whole image
#     bins, values = value_hists(rgb_image, 'rgb')
#     if plot:
#         plotHist(bins, values, 6, 'rgb', 'value', False)
    # Convert absimage to HSV color space
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    
    image = np.copy(hsv)
    # green
    lower_green = np.array([80,120,80]) 
    upper_green = np.array([140,255,255])
    #yellow
    lower_yellow = np.array([0,100,120]) 
    upper_yellow = np.array([80,255,255])
    #red
    lower_red = np.array([140,50,20]) 
    upper_red = np.array([180,255,255])
    
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    masked_image = np.zeros((image.shape[1],image.shape[0],3), np.uint8)
    masked_image[mask_red != 0] = [255, 0, 0]
    masked_image[mask_green != 0] = [0, 255, 0]
    masked_image[mask_yellow != 0] = [0, 0, 255]
        
    return masked_image

# #image, label, image_num = getImage(color='red', Random=True, showImg=False)
# #image_num = random.randint(0, total_num_of_images-1)
# image_num = 757
# test_im = STANDARDIZED_LIST[image_num][0]
# test_label = STANDARDIZED_LIST[image_num][1]

feature = feature_rgb(test_im, True)
print(test_label)
plt.imshow(feature)
plt.show()
plt.imshow(test_im)
plt.show()


# ## (QUESTION 1): How do the features you made help you distinguish between the 3 classes of traffic light images?

# **Answer:**
# * the first feature detects brightness per row, along the length of the traffic light. The mean brightness can be found and charaterized as red, yellow, or green based on where it is in the image
# * the second feature multiples the hue and value histograms together to reinforce or question the value filter.
# * the third feature filters the image by hue color. This can be used to see if any red, yellow, or green pixels have been found, and which is the most dominant in the image. 

# # 4. Classification and Visualizing Error
# 
# Using all of your features, write a function that takes in an RGB image and, using your extracted features, outputs whether a light is red, green or yellow as a one-hot encoded label. This classification function should be able to classify any image of a traffic light!
# 
# You are encouraged to write any helper functions or visualization code that you may need, but for testing the accuracy, make sure that this `estimate_label` function returns a one-hot encoded label.

# ---
# <a id='task8'></a>
# ### (IMPLEMENTATION): Build a complete classifier 

# In[70]:


# This function should take in RGB image input
# Analyze that image using your feature creation code and output a one-hot encoded label

#boolean flag if hot encoder is double encoded
def double_encoded(one_hot_encoded):
    if (np.sum(one_hot_encoded) > 1):
        return True
    else:
        return False
    
# boolean flag if hot encoder has a label
def has_label(one_hot_encoded):
    if (np.sum(one_hot_encoded) > 0):
        return True
    else:
        return False

# hot encode the image based on where the mean of the feature is in relation to the height of the
# traffic light.
def hot_encode_height(mean):
    # bin ranges
    # red = 0, 15
    # yellow = 10, 23
    # green = 20, 32
    one_hot_encoded = [0, 0, 0]
    
    if mean >= 0 and mean < 15:
        one_hot_encoded[0] = 1
    if mean >= 10 and mean < 23:
        one_hot_encoded[1] = 1
    if mean >= 20 and mean <= 32:
        one_hot_encoded[2] = 1
    
    return one_hot_encoded

def estimate_value(rgb_image):
    
    feature = feature_value(rgb_image)
    
    # get mean and bimodal boolean
    max_list2 = max_idx_rank(feature)
    mean = max_list2[0]
    bimodal = is_bimodal(max_list2, feature)
    
    one_hot_encoded = hot_encode_height(mean)
    return one_hot_encoded, bimodal 


def estimate_hueXvalue(rgb_image):
    
    feature = feature_valueXHue(rgb_image)
    
    # get mean and bimodal boolean
    max_list2 = max_idx_rank(feature)
    mean = max_list2[0]
    bimodal = is_bimodal(max_list2, feature)
    
    one_hot_encoded = hot_encode_height(mean)
    return one_hot_encoded, bimodal 
        
    
def estimate_color(rgb_image):
    feature = feature_rgb(rgb_image)
    
    one_hot_encoded = [0, 0, 0]
    # sum channels representing each light color
    red_sum = np.sum(feature[:,:,0])
    green_sum = np.sum(feature[:,:,1])
    yellow_sum = np.sum(feature[:,:,2])
    
    # one hot encode the color who has the greatest sum
    if red_sum > (yellow_sum + green_sum):
        one_hot_encoded[0] = 1
    if yellow_sum > (green_sum + red_sum):
        one_hot_encoded[1] = 1
    if green_sum > (yellow_sum + red_sum):
        one_hot_encoded[2] = 1
        
    return one_hot_encoded
        
# rules
    # * if all hues detected, or white is predominant. color is most liekly yellow.
    # * if a bimodal distribution of value vs traffic light length is found.
    #     - compute hue * value.
    #     - use hue * value mean retrieved from this plot.

def estimate_label(rgb_image, print_stats = False):
    
    ## TODO: Extract feature(s) from the RGB image and use those features to
    ## classify the image and output a one-hot encoded label
    predicted_label = [0,0,0]
    predicted_label_height = []
    predicted_label_color = []
    bimodal = False
    # copy picture
    image = np.copy(rgb_image)
    
    # get hot encode for value vs height
    predicted_label_Value, bimodal_value = estimate_value(image)
    
    if print_stats:
        print('height: {}, bimodal: {}'.format(predicted_label_Value, bimodal_value))
    
    # if bimodal compute hue * value and get the hot encode vs height
    if bimodal == True:
        predicted_label_height, bimodal = estimate_hueXvalue(image)
    else:
        predicted_label_height = predicted_label_Value
        bimodal = bimodal_value
    
    # get hot encode for color
    predicted_label_color = estimate_color(image)
    
    if print_stats:
        print('height: {}, bimodal: {}'.format(predicted_label_height, bimodal))
        print('color: {}'.format(predicted_label_color))
    
    #union of two hot_encoded labels
    union = matrix_multiplication(predicted_label_height, predicted_label_color)
    
    #if height label and color label match return encoder
    if predicted_label_height == predicted_label_color:
        predicted_label = predicted_label_height
        # check to make sure two colors arn't labeled
        if not double_encoded(predicted_label_height):
            return predicted_label
        
    # if height label and color label do not match
    #if double encoded
    if double_encoded(predicted_label_height):
        #use color if not double encoded
        if not double_encoded(predicted_label_color) and has_label(predicted_label_color):
            # check to make sure estimates do not conflict, and contain a union
            if np.sum(union) > 0:
                predicted_label = predicted_label_color
            else:
                #if double encoded on green or red choose red
                if predicted_label_height[0]*predicted_label_height[1] > 0:
                    predicted_label = [1, 0, 0]
            return predicted_label
        #if color is double encoded
        else:
            # if they are encoded the same choose red if not use the union
            if np.sum(union) == 1:
                predicted_label = union
            else:
                predicted_label = [1, 0, 0]
            return predicted_label
    # if height label is not double labelled and conflicts with color
    # choose color if available
    else:        
        if has_label(predicted_label_color) and predicted_label_color != [0, 1, 0]:
            predicted_label = predicted_label_color
        # if no color got off of value feature
        # if bimodal 
        else:
            if bimodal_value:
                predicted_label = predicted_label_height
            else:
                predicted_label = predicted_label_Value
    
    return predicted_label   

image, label, image_num = getImage(color='red', Random=True, showImg=False)
#image_num = random.randint(0, total_num_of_images-1)
#image_num = 757
test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]

encoder = estimate_label(test_im)
print('image: ', image_num)
print('guess: ', encoder)
print('actual: ', test_label)
plt.imshow(test_im)
plt.show()


# ## Testing the classifier
# 
# Here is where we test your classification algorithm using our test set of data that we set aside at the beginning of the notebook! This project will be complete once you've pogrammed a "good" classifier.
# 
# A "good" classifier in this case should meet the following criteria (and once it does, feel free to submit your project):
# 1. Get above 90% classification accuracy.
# 2. Never classify a red light as a green light. 
# 
# ### Test dataset
# 
# Below, we load in the test dataset, standardize it using the `standardize` function you defined above, and then **shuffle** it; this ensures that order will not play a role in testing accuracy.
# 

# In[20]:


# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)


# ## Determine the Accuracy
# 
# Compare the output of your classification algorithm (a.k.a. your "model") with the true labels and determine the accuracy.
# 
# This code stores all the misclassified images, their predicted labels, and their true labels, in a list called `MISCLASSIFIED`. This code is used for testing and *should not be changed*.

# In[94]:


# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)

def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))


# ---
# <a id='task9'></a>
# ### Visualize the misclassified images
# 
# Visualize some of the images you classified wrong (in the `MISCLASSIFIED` list) and note any qualities that make them difficult to classify. This will help you identify any weaknesses in your classification algorithm.

# In[92]:


# Visualize misclassified example(s)
## TODO: Display an image in the `MISCLASSIFIED` list 
## TODO: Print out its predicted label - to see what the image *was* incorrectly classified as
rand_misclass_idx = random.randint(0, len(MISCLASSIFIED)-1)
rand_misclass_idx = 17
test_im = MISCLASSIFIED[rand_misclass_idx][0]
test_label = MISCLASSIFIED[rand_misclass_idx][2]
encoder = MISCLASSIFIED[rand_misclass_idx][1]

encoder = estimate_label(test_im, True)
print('guess: ', encoder)
print('actual: ', test_label)
plt.imshow(test_im)
plt.show()


# ---
# <a id='question2'></a>
# ## (Question 2): After visualizing these misclassifications, what weaknesses do you think your classification algorithm has? Please note at least two.

# **Answer:** if light has not been cropped at the top, and there is a siginificant brightness at the top, for the height feature it will distinguish this in the value plot. THe height feature has a significant error probablity, where the hue feature was used as validation. However the hue ranges between the three colors are not  always distint, and cannot be filtered easily. greater precision in the hue filters would be needed to increase the accuracy of this model. 

# ## Test if you classify any red lights as green
# 
# **To pass this project, you must not classify any red lights as green!** Classifying red lights as green would cause a car to drive through a red traffic light, so this red-as-green error is very dangerous in the real world. 
# 
# The code below lets you test to see if you've misclassified any red lights as green in the test set. **This test assumes that `MISCLASSIFIED` is a list of tuples with the order: [misclassified_image, predicted_label, true_label].**
# 
# Note: this is not an all encompassing test, but its a good indicator that, if you pass, you are on the right track! This iterates through your list of misclassified examples and checks to see if any red traffic lights have been mistakenly labelled [0, 1, 0] (green).

# In[95]:


# Importing the tests
import test_functions
tests = test_functions.Tests()

if(len(MISCLASSIFIED) > 0):
    # Test code for one_hot_encode function
    tests.test_red_as_green(MISCLASSIFIED)
else:
    print("MISCLASSIFIED may not have been populated with images.")


# # 5. Improve your algorithm!
# 
# **Submit your project after you have completed all implementations, answered all questions, AND when you've met the two criteria:**
# 1. Greater than 90% accuracy classification
# 2. No red lights classified as green
# 
# If you did not meet these requirements (which is common on the first attempt!), revisit your algorithm and tweak it to improve light recognition -- this could mean changing the brightness feature, performing some background subtraction, or adding another feature!
# 
# ---

# ### Going Further (Optional Challenges)
# 
# If you found this challenge easy, I suggest you go above and beyond! Here are a couple **optional** (meaning you do not need to implement these to submit and pass the project) suggestions:
# * (Optional) Aim for >95% classification accuracy.
# * (Optional) Some lights are in the shape of arrows; further classify the lights as round or arrow-shaped.
# * (Optional) Add another feature and aim for as close to 100% accuracy as you can get!
