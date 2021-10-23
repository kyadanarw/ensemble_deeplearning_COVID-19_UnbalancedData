# Ensemble-Deep-Learning-for-the-Detection-of-COVID-19-in-Unbalanced-Chest-X-ray-Dataset
We demonstrates ensemble deep learning to classify COVID-19, viral pneumonia and normal on unbalanced chest X-rays dataset.
Initially, we preprocessed and segmented the lung regions usingDeepLabV3+ method, and subsequently cropped the lung regions. The cropped lung regions were used as inputs to several deep convolutional neural networks (CNNs) for the prediction of COVID-19. The dataset was highly unbalanced; the vast majority were normal images, with a small number of COVID-19 and pneumonia images. To remedy the unbalanced distribution and to avoid biased classification results, we applied five different ap-proaches: (i) balancing the class using weighted loss; (ii) image augmentation to add more images to minority cases; (iii) the undersampling of majority classes; (iv) the oversampling of minority classes; and (v) a hybrid resampling approach of over-sampling and undersampling. The best-performing methods from each approach were combined as the ensemble classifier using two voting strategies. Finally, we used the saliency map of CNNs to identify the indicative regions of COVID-19 biomarkers which are deemed useful for interpretability.
This repository contains matlab and python files.
Matlab files contain source code for image preprocessing, lung segmentation, data partition and data augmentation.
Python files contain source code for fine tuning the pretrained CNNs with different approaches to handle the unbalanced class distribution.
Apporach_0: fine tuning the pretrained models with categorial cross-entopy loss.
Approach_1: fine tuning the pretrained models using the weighted cross-entropy loss to handle the unbalanced class distribution.
Approach_2: fine tuning the pretrained models using the image augmentation to handle the unbalanced class distribution.
Approach_3: fine tuning the pretrained models using the undersampling to handle the unbalanced class distribution.
Approach_4: fine tuning the pretrained models using the oversampling to handle the unbalanced class distribution.
Approach_5: fine tuning the pretrained models using the hybrid sampling to handle the unbalanced class distribution. We also demonstrate ensemble learning of deep CNNs using majority hard and soft voting strategies.
