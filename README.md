# Mitochondria-Image-Segmentation
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
[]!(./images/Figure1.png)
