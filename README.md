# Ensemble-Deep-Learning-for-the-Detection-of-COVID-19-in-Unbalanced-Chest-X-ray-Dataset
<p align="justify">The goal of this project is to classify the COVID-19 from normal and pneumonia on Chest X-rays in unbalanced data distribution setting.         
How I proceeded exactly and what results I achieved can be read in my paper. https://doi.org/10.3390/app112210528</p>

## Overview
 <p align="justify">
       We demonstrates ensemble deep learning to classify COVID-19, viral pneumonia and normal on unbalanced chest X-rays dataset. Initially, we preprocessed andsegmented the lung regions usingDeepLabV3+ method, and subsequently cropped the lung regions. The cropped lung regions were used as inputs to several deep convolutional neural networks (CNNs) for the prediction of COVID-19. The dataset was highly unbalanced; the vast majority were normal images, with a small number of COVID-19 and pneumonia images. To remedy the unbalanced distribution and to avoid biased classification results, we applied five different ap-proaches:     (i) balancing the class using weighted loss;    (ii) image augmentation to add more images to minority cases;    (iii) the undersampling of majority classes;    (iv) the oversampling of minority classes; and     (v) a hybrid resampling approach of over-sampling and undersampling. The best-performing methods from each approach were combined as the ensemble classifier using two voting strategies. Finally, we used the saliency map of CNNs to identify the indicative regions of COVID-19 biomarkers which are deemed useful for interpretability.
</p>

## Table of contents
* [Introduction ](#introduction) 
* [Software Requirements](#software-requirements) 
* [Data](#data) 
* [Getting Started](#getting-started)
* [Folder Structure](#folder-structure)
* [Running Matlab files](#running-matlab-files)
* [Running the Jupyter Notebooks](#running-the-jupyter-notebooks)
* [Classify Chest Xrays Images](#classify-chest-xrays-images)
* [Project Results](#project-results)
* [Link to the Publication](#link-to-the-publication)
* [Authors](#authors)
* [Project Motivation](#project-motivation)
* [References](#references)

## Introduction 
<p align="justify"> This repository contains matlab and python files. Matlab files contain source code for image preprocessing, lung segmentation, data partition and data augmentation.Python files contain source code for fine tuning the pretrained CNNs with different approaches to handle the unbalanced class distribution. 
</p>

* The image preprocessing is performed using meidan filter and adaptive histogram equalization to filter the noises and enahnce the contrast of the images (Preprocessing_Images.m).
* DeeplabV3+ with Xception backbone is used as a semantic segmentation algorithm to segregate the lung regions from chest x-rays (Train_LungSegmentation_DeepLab.m). 
* The image files are randomly partitioned into train, test and validation for model training and testing.

* Apporach_0: fine tuning the pretrained models with categorial cross-entopy loss.
* Approach_1: fine tuning the pretrained models using the weighted cross-entropy loss to handle the unbalanced class distribution.
* Approach_2: fine tuning the pretrained models using the image augmentation to handle the unbalanced class distribution.
* Approach_3: fine tuning the pretrained models using the undersampling to handle the unbalanced class distribution.
* Approach_4: fine tuning the pretrained models using the oversampling to handle the unbalanced class distribution.
* Approach_5: fine tuning the pretrained models using the hybrid sampling to handle the unbalanced class distribution. We also demonstrate ensemble learning of deep CNNs using majority hard and soft voting strategies.

## Software Requirements
Required libraries:
* Matlab 2020A and later versions
* Python 3.x
* Scikit-Learn
* Keras
* TensorFlow
* Numpy
* Pandas
* Matplotlib
* OpenCV
## Data
we developed and trained DeepLabV3+-based lung segmentation using a combined dataset from Montgomery (MC), Shenzhen, and Japanese Society of Radiological Technology (JSRT) databases. The COVID-19 Radiography Database is used for classification of COVID-19 from normal and pneumonia chest x-rays.
* Montgomery Dataset
* Shenzhen Dataset 
* Japanese Society of Radiological Technology (JSRT) Dataset
* the COVID-19 Radiography Database https://www.kaggle.com/tawsifurrahman/covid19-radiography-database 

## Getting Started
###Make sure Matlab and Python 3 is installed.
Clone the repository and navigate to the project's root directory in the terminal
Download the cats dataset. Unzip the folder and place the images in the cloned repository in the folder cats. If the folder does not exist yet, please create one.
Download the dogs dataset. Unzip the folder and place the images in the cloned repository in the folder dogs. If the folder does not exist yet, please create one.
Download the wilds dataset. Unzip the folder and place the images in the cloned repository in the folder wilds. If the folder does not exist yet, please create one.
Start the notebook CNN with Transfer Learning for Multi-Class Classification.ipynb.

## Folder Structure

## Running Matlab files
Make sure Matlab 2020a or later version is installed.

## Running the Jupyter Notebooks

## Classify Chest Xrays Images
<p align="justify">Predict the test data using this code.</p>

![Predict](https://github.com/kyadanarw/Ensemble-Deep-Learning-for-the-Detection-of-COVID-19-in-Unbalanced-Chest-X-ray-Dataset/blob/DeepLearning/Images/Preddict_new_images.png)

<p align="justify">Plot ROC curve using the following code, make sure to call get_roc_curve function to your notebook.</p>

![PlotROC](https://github.com/kyadanarw/Ensemble-Deep-Learning-for-the-Detection-of-COVID-19-in-Unbalanced-Chest-X-ray-Dataset/blob/DeepLearning/Images/Plot_ROC_curve.png)

<p align="justify">Evaluate the performance of the algorithm with different performance measures using the following code.</p>

![Evaluation](https://github.com/kyadanarw/Ensemble-Deep-Learning-for-the-Detection-of-COVID-19-in-Unbalanced-Chest-X-ray-Dataset/blob/DeepLearning/Images/EvaluationMetrics.png)

<p align="justify">Visualize the indicative regions of the COVID_19 using the following code.</p>

![Grad-CAM](https://github.com/kyadanarw/Ensemble-Deep-Learning-for-the-Detection-of-COVID-19-in-Unbalanced-Chest-X-ray-Dataset/blob/DeepLearning/Images/GradCam_Map.png)

## Project Results
<p align="justify">Lung segementation using deeplabv3+ and Xception model.</p>

![Segmentation](https://github.com/kyadanarw/Ensemble-Deep-Learning-for-the-Detection-of-COVID-19-in-Unbalanced-Chest-X-ray-Dataset/blob/DeepLearning/Images/applsci-11-10528-g005-550.jpg)

<p align="justify">We achieved the highest accuracy of 99.23% and an AUC of 99.97% using an ensemble classifier of XceptionNet, MobileNetV2, DensetNet201, InceptionResNetV2 and NasNetMobile with image augmentation.</p>

![EnsembleResult](https://github.com/kyadanarw/Ensemble-Deep-Learning-for-the-Detection-of-COVID-19-in-Unbalanced-Chest-X-ray-Dataset/blob/DeepLearning/Images/Ensemble%20Results.png)

<p align="justify">ROC curves of ensemble classifiers of different apporaches.</p>

![ROC_Curve](https://github.com/kyadanarw/Ensemble-Deep-Learning-for-the-Detection-of-COVID-19-in-Unbalanced-Chest-X-ray-Dataset/blob/DeepLearning/Images/applsci-11-10528-g016-550.jpg)

<p align="justify">The indicative regions of COVID-19 biomarkers highlighted by DenseNet201.</p>

![HeatMap](https://github.com/kyadanarw/Ensemble-Deep-Learning-for-the-Detection-of-COVID-19-in-Unbalanced-Chest-X-ray-Dataset/blob/DeepLearning/Images/applsci-11-10528-g017-550.jpg)


## Link to the Publication
https://doi.org/10.3390/app112210528

## Authors
* Khin Yadanar Win
* Syna Sreng

## Project Motivation
<p align="justify"> I've been working on medical image analysis using image processing and machine learning since 2015. The list of my publications can be found on https://www.researchgate.net/profile/Khin-Win-13. I also publish individual interesting sections from my publications in separate repositories to make their access even easier. </p>

## References
The full list of the references can be found in the paper.
Transfer learning .ipynb files are inspired by Coursera's AI for Medicine course. 
