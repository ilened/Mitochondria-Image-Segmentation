# Mitochondrion Image Segmentation
## Introduction
For our project, we learned about image segmentation through various methods readily available in the field. Matlab has a large Image Processing Toolbox filled with many tools and methods for use in image analysis. We chose three different methods to compare on one, central data set.

## Motivation
In the field of biology and medicine, it is important for scientists and researchers to analyze microscope slides collected. Often, scientists want to isolate cells and their distinct structures. However, analyzing this by hand can be very tedious. To analyze these cells computationally, one must first segment the images for later object recognition. 

## Objective 
For our project, we will tackle the important preprocessing step of image segmentation. Our objective is to segment cells and their parts–in this case, mitochondria. We are concerned with finding an optimization method that yields the best cell segmentation. 

We will segment these images using three methods within the Matlab Image Processing Toolbox:
- K-means clustering
- Sobel edge detection
- Canny edge detection  

We will compare these methods based on their ability to properly segment our data set quantitatively. By means of hyperparameter optimization, we are trying to find the best combination of parameters to get the best performance of each method.

## Dataset
For our data set, we used a Focused Ion Beam Scanning Electron Microscopy (FIB-SEM) image stack (“Electron Microscopy Dataset”). This dataset consists of cells and their mitochondria. We chose this dataset because segmenting cells and their parts is an important feat. Also, this dataset provides a challenge since the mitochondria are a similar color to the cell walls, and the cell walls are the same color as the mitochondria walls. Below is an example of an original image and its corresponding ground truth segmentation of its mitochondria.
![](./images/Figure1.png)

## Methods
Below, we delineate three methods we chose to compare: Sobel edge detection, Canny edge detection, and K-means clustering. We validated each algorithm on a set of 20 “training” images. We put training in quotes because we are not training our algorithms here. We are implementing the algorithm and tuning parameters in order to minimize the error between the resulting segmented image and the ground truth segmentation. We explain error in our Results section.

### Sobel Edge Detection

### *Background*
This method involves computing the gradient of the image for each pixel position in the image. A pixel location is declared an edge location if the value of the gradient exceeds some threshold. Edges have higher pixel intensity values than those intensity values surrounding it. So once a threshold is set, we can compare the gradient value to the threshold value and detect an edge whenever the threshold is exceeded.

The Sobel operator performs a 2-D spatial gradient measurement on an image and emphasizes regions of high spatial gradient that correspond to edges. Typically, it is used to find the approximate absolute gradient magnitude at each point in an input grayscale image. Compared to other edge operator, Sobel has two main advantages:

1) With the introduction of the average factor, it has a smoothing effect on the random noise of the image.
2) The elements of the edge on both sides has been enhanced, so the edge seems thick and bright. (Gupta)

### *Implementation*
To implement this method in Matlab, we did the following:
1) Turn the image into binary by threshold (multiplied by a fudge factor)
2) Complement/reverse of the binary image
3) Use Matlab’s edge(A, ‘sobel’) function to detect edges in image A. This method also returns a preliminary threshold for which to run Sobel again.
4) Dilate the Image (Connect the segmented pieces) using Matlab’s imdilate(function)
5) Fill Interior Gaps using imfill() function
6) Remove connected objects on Border using imclearboarder() function
7) Smoothen the objects by eroding thin lines twice
8) Remove objects smaller than minPixels; which we set to 150

### *Tuning*
