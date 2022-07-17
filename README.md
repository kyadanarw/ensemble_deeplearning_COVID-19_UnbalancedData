# Ensemble-Deep-Learning-for-the-Detection-of-COVID-19-in-Unbalanced-Chest-X-ray-Dataset
<p align="justify">The goal of this project is to classify the COVID-19 from normal and pneumonia on Chest X-rays in unbalanced data distribution setting.         
This GitHub repository serves as the source code for a published paper in https://doi.org/10.3390/app112210528</p>



## Overview
 <p align="justify">
       We demonstrates ensemble deep learning to classify COVID-19, viral pneumonia and normal on unbalanced chest X-rays dataset. Initially, we preprocessed andsegmented the lung regions usingDeepLabV3+ method, and subsequently cropped the lung regions. The cropped lung regions were used as inputs to several deep convolutional neural networks (CNNs) for the prediction of COVID-19. The dataset was highly unbalanced; the vast majority were normal images, with a small number of COVID-19 and pneumonia images. To remedy the unbalanced distribution and to avoid biased classification results, we applied five different ap-proaches:     (i) balancing the class using weighted loss;    (ii) image augmentation to add more images to minority cases;    (iii) the undersampling of majority classes;    (iv) the oversampling of minority classes; and     (v) a hybrid resampling approach of over-sampling and undersampling. The best-performing methods from each approach were combined as the ensemble classifier using two voting strategies. Finally, we used the saliency map of CNNs to identify the indicative regions of COVID-19 biomarkers which are deemed useful for interpretability.
</p>

## Table of contents
* [Introduction ](#introduction) 
* [Software Requirements](#software-requirements) 
* [Data](#data) 
* [Getting Started](#getting-started)
* [Running Matlab files](#running-matlab-files)
* [Running Python Jupyter Notebooks](#running-python-jupyter-notebooks)
* [Classify Chest Xrays Images](#classify-chest-xrays-images)
* [Project Results](#project-results)
* [Link to the Publication](#link-to-the-publication)
* [Authors](#authors)
* [Project Motivation](#project-motivation)
* [References](#references)

## Introduction 
<p align="justify"> This repository contains matlab and python files. Matlab files contain source code for image preprocessing, lung segmentation, data partition and data augmentation.Python files contain source code for fine tuning the pretrained CNNs. The following six different approaches are compared to handle the unbalanced class distribution. </p>

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
We developed and trained DeepLabV3+-based lung segmentation using a combined dataset from Montgomery (MC), Shenzhen, and Japanese Society of Radiological Technology (JSRT) databases. The COVID-19 Radiography Database is used for classification of COVID-19 from normal and pneumonia chest x-rays.
* Montgomery Dataset
* Shenzhen Dataset 
* Japanese Society of Radiological Technology (JSRT) Dataset
* the COVID-19 Radiography Database https://www.kaggle.com/tawsifurrahman/covid19-radiography-database 

## Getting Started
#### Make sure Matlab and Python 3 is installed.

 1. Clone the repository and navigate to the project's root directory in the terminal
2. Download the mongometry, shenzen and jsrt dataset. Unzip the folder and place original images and their groundtruth label in seperate folders.
3. Start Train_LungSegmentation_DeepLab.m for lung segmentation. 
4. Download the COVID-19 Radiography dataset. Unzip the folder and place the images in the cloned repository in another folder. If the folder does not exist yet, please create one.
5. Run Preprocessing_Images.m and Test_LungSegmentation.m to segment the lung and crop the lung regions of the COVID-19 X-rays dataset and place the cropped X-rays images in one folder. If the folder does not exist, please create one.
6. Run the python notebooks.


## Running Matlab files
* First run the Train_LungSegmentation_DeepLab.m to train the Deeplabv3 based semantic segmentation model using the combined dadataset of mongomery, shenzen and JSRT.<br/>
* Save the trained semantic segmentation model.<br/>
* Then run Test_LungSegmentation.m to segment and crop the lung regions from chest x-rays of COVID-19 dataset and place the cropped images in one folder.
Here are the results of the lung segmentation, first image is semantic label, second one is binary map of segmented lung RIO and the last one is cropped lung RIO:

![Semantic](https://github.com/kyadanarw/Ensemble-Deep-Learning-for-the-Detection-of-COVID-19-in-Unbalanced-Chest-X-ray-Dataset/blob/DeepLearning/Images/semantic56.jpg)

![BinaryMap](https://github.com/kyadanarw/Ensemble-Deep-Learning-for-the-Detection-of-COVID-19-in-Unbalanced-Chest-X-ray-Dataset/blob/DeepLearning/Images/lung56.jpg) 

![CroppedLung](https://github.com/kyadanarw/Ensemble-Deep-Learning-for-the-Detection-of-COVID-19-in-Unbalanced-Chest-X-ray-Dataset/blob/DeepLearning/Images/lung56_pixels.jpg)

Run Preprocessing_Images.m to improve the quality of the image before inputing into deep learning for prediction of COVID-19.

## Running Python Jupyter Notebooks
Import all necessary libraries and execute all .ipynb files and save the trained models. <br/>
Here is the code to train the model:

![TrainModel](https://github.com/kyadanarw/Ensemble-Deep-Learning-for-the-Detection-of-COVID-19-in-Unbalanced-Chest-X-ray-Dataset/blob/DeepLearning/Images/train_model.png)

Here is the output of the function:

![TrainHist](https://github.com/kyadanarw/Ensemble-Deep-Learning-for-the-Detection-of-COVID-19-in-Unbalanced-Chest-X-ray-Dataset/blob/DeepLearning/Images/train_his.png)

![TrainHistoryLoss](https://github.com/kyadanarw/Ensemble-Deep-Learning-for-the-Detection-of-COVID-19-in-Unbalanced-Chest-X-ray-Dataset/blob/DeepLearning/Images/training_history.png)
<br/>
Save the trained models.<br/>
Run EnsembleLearning-HardVoting.ipynb to build an hard-voting based ensemble of CNNs with different approaches.<br/>
Run EnsembleLearning-SoftVoting.ipynb to build an soft-voting based ensemble of CNNs with different approaches.<br/>

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

## Citation
Win, Khin Y., Noppadol Maneerat, Syna Sreng, and Kazuhiko Hamamoto. 2021. "Ensemble Deep Learning for the Detection of COVID-19 in Unbalanced Chest X-ray Dataset" Applied Sciences 11, no. 22: 10528. https://doi.org/10.3390/app112210528 

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
