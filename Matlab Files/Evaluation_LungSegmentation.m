close all
clear all
clc

%%
%load the trained semantic model
net=load('DeepLab_Xception.mat');
net=net.net;

%%
%load the dataset
dataSetDir = fullfile('Test_Images');

testImagesDir = fullfile(dataSetDir,'TestingImages');
testimds = imageDatastore(testImagesDir);

testLabelsDir = fullfile(dataSetDir,'TestingLabels');
test_label = imageDatastore(testLabelsDir);

%%
%READ the image files
TestImage=testimds.Files;
TestLabel=test_label.Files;

%%
%construct the array to store the evaluation results
scoreD=[];
scoreJ=[];
scoreB=[];
scoreAcc=[];
count=length(TestImage);


%%
%Evaluate all images in the testing folder
for k = 1:count
    img=cell2mat(TestImage(k));
    testImage=imread(img);
    C = semanticseg(testImage,net);
    Mask = C == 'Lung'; %%generate the lung Mask

   imglabel=cell2mat(TestLabel(k));
   GT=imread(imglabel);
   GT=logical(GT); %%Ground truth Mask
   
   
   dsc_score=dice(Mask,GT);  %%dice similarity score
   scoreD=[scoreD,dsc_score];
   
   jsc_score= jaccard(Mask,GT);  %jaccard index or IoU
   scoreJ=[scoreJ,jsc_score];
   
   bfs_score=bfscore(Mask,GT); %%BFS score
   scoreB=[scoreB,bfs_score];
   
   accuracy = mean(Mask == GT);  %%Accuracy
   accuracy=mean(accuracy);
   scoreAcc=[scoreAcc,accuracy];
end
   scoreD=transpose(scoreD);
   mean_DSC=mean(scoreD);
   std_DSC=std(scoreD);
   
   scoreJ=transpose(scoreJ);
   mean_JSC=mean(scoreJ);
   std_JSC=std(scoreJ)
   
   scoreB=transpose(scoreB);
   mean_BSF=mean(scoreB);
   std_BSF=std(scoreB);
   
   scoreAcc=transpose(scoreAcc);
   mean_Acc=mean(scoreAcc);
   std_Acc=std(scoreAcc);

   Results=[mean_JSC,mean_DSC,mean_Acc,mean_BSF];
   format short
   Results_Std=[std_JSC,std_DSC,std_Acc,std_BSF];