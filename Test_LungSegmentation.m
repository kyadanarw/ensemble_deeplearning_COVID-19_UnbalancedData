close all
clear all
clc

%%
%load the trainedNet
net=load('DeepLab_Xception.mat');
net=net.net;

%%
%load the testing image
testImage = imread('text_X_rays\t001.png');


%%
%Generate the Lung Mask using trained Net
C = semanticseg(testImage,net);

B = labeloverlay(testImage,C);
figure
imshow(B)

Extract_LungMask= C == 'Lung';

%%
%applymmorpholocal operations for close the holes
se = strel('disk',10)
lungMask=imclose(Extract_LungMask,se)

%%
%reconstruct images into original form
maskedRgbImage = bsxfun(@times, testImage, cast(LungMask, 'like', testImage));


%%
%%Cropping Lung regions
r=maskedRgbImage(:,:,1);
g=maskedRgbImage(:,:,2);
b=maskedRgbImage(:,:,3);
col1 = find(sum(lungMask, 1), 1, 'first');
col2 = find(sum(lungMask, 1), 1, 'last');
ro1 = find(sum(lungMask, 2), 1, 'first');
ro2 = find(sum(lungMask, 2), 1, 'last');
RSegmented=r(ro1:ro2, col1:col2);
GSegmented=g(ro1:ro2, col1:col2);
BSegmented=b(ro1:ro2, col1:col2);
Cropped_RGB_Lung=cat(3,RSegmented,GSegmented,BSegmented);


