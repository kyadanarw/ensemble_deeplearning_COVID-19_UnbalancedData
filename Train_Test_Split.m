close all
close all
clear all
clc
tic
rng(1)
%%
%create the datastore
imds=imageDatastore(fullfile('C:\Users\CroppedLungXrays\'),...
'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames');

%%
%split the dataset into traning and testing sets in 80 and 20%
[imdsTrain_Validate,imdsTest] = splitEachLabel(imds,0.8,'randomized');
[imdsTrain,imdsValidate] = splitEachLabel(imdsTrain_Validate,0.8,'randomized');


Train_Files=imdsTrain.Files;
Test_Files=imdsTest.Files;

Train_Labels=imdsTrain.Labels;
Test_Labels=imdsTest.Labels;


Validate_Files=imdsValidate.Files;
Validate_Files=imdsValidate.Files;


Train_data=[Train_Files, Train_Labels];
Test_data=[Test_Files,Test_Labels];
Validate_data=[Validate_Files,Validate_Labels];


csvwrite('Train_Data.csv',Train_data)
csvwrite('Test_Data.csv',Test_data)
csvwrite('Validate_Data.csv',Test_data)