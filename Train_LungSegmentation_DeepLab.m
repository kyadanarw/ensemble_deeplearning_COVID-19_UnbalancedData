
close all
clear all
clc

imageSize = [512 512];
numClasses = 2;
network = 'xception';
lgraph = deeplabv3plusLayers(imageSize,numClasses,network, ...
             'DownsamplingFactor',16);
         analyzeNetwork(lgraph)
       

dataSetDir = fullfile('lung_X_rays');
imageDir = fullfile(dataSetDir,'TrainImages');
labelDir = fullfile(dataSetDir,'TrainLabels');

imds = imageDatastore(imageDir);

classNames = ["Lung","Background"];
labelIDs   = [255 0];
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);

imageSize = [512 512];
numClasses = numel(classNames);
lgraph = deeplabv3plusLayers(imageSize,numClasses,'xception');

pximds = pixelLabelImageDatastore(imds,pxds,'OutputSize',imageSize,...
    'ColorPreprocessing','gray2rgb');

opts = trainingOptions('adam',...  %%rmsprop, sgdm
    'MiniBatchSize',2,...
    'MaxEpochs',3,...
        'ExecutionEnvironment','gpu',...
    'Plots','training-progress');

net = trainNetwork(pximds,lgraph,opts);

save net