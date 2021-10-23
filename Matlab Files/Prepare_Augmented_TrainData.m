close all
clear all
clc

%%
%Retrive the augmented image files of COVID-19 CXRs
myFolder = 'C:\Users\AugmentedImages\COVID-19\';
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end

%get the file names 
filePattern = fullfile(myFolder, '*.png');
Files = dir(filePattern);
Aug_Covid_Files=Files('name')
n=lenght(Aug_Covid_Files);
label_covid=zeros(n,1);
label_covid=str(label_covid);
Label_Covid= strrep(label_covid,'0','COVID-19')
COVID_Aug=[Aug_Covid_Files,Label_Covid];


%%
%Retrive the augmented image files of pneumonia CXRs
myFolder = 'C:\Users\AugmentedImages\Viral Pneumonia\';
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
%get the file names 
filePattern1 = fullfile(myFolder, '*.png');
Files1 = dir(filePattern1);
Aug_Pneu_Files=Files1('name')
n=lenght(Aug_Pneu_Files);
label_P=zeros(n,1);
label_P=str(label_P);
Label_Pneumonia= strrep(label_P,'0','Viral Pneumonia')
Pneumonia_Aug=[Aug_Pneu_Files,Label_Pneumonia];

%load the original trained dataset
trained_data_origin=load('Trained_data.mat');

%concatenate the original training data and augmented data
Augmented_Train_Data=[trained_data_origin;COVID_Aug;Pneumonia_Aug];

%Save the augmented trained data file as csv file
csvwrite('Augmented_Train_Data.csv',Augmented_Train_Data)
