clc
clear all
close all


myFolder = 'C:\Users\Chest-X-rays\';
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end

filePattern = fullfile(myFolder, '*.png');
jpegFiles = dir(filePattern);
for k = 1:length(jpegFiles)
  baseFileName = jpegFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  imageArray = imread(fullFileName);
  
filtered_img=medfilt2(imageArray)

enhanced_img=adapthisteq(filtered_img,'clipLimit',0.0065,'Distribution','rayleigh');


 savefileh=['C:\enhanced_X-rays\',baseFileName];
 imwrite(enhanced_img,savefileh);

end