close all
clear all
clc


%load the data that are required image augmentation
c= readtable('Trained_Data.csv',...
    'Delimiter','_','ReadVariableNames',false)


%naviate the image folder
myFolder = 'C:\Users\Images\';
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end

d=c{:,:};
n=length(d)

for k = 1:n
  baseFileName = d(k)
  fullFileName = fullfile(myFolder, baseFileName);
  fullFileName=char(fullFileName);
  
  I= imread(fullFileName);
  [x,y,z]=size(I)
  if z<3
      I=cat(3,I,I,I);
  end    
  %%
  %flip
  F = flipdim(I ,2);   
  F=imresize(F,[512 512]);
 
  savefile_f=['C:\Users\\AugmentedImages\Flip_',char(baseFileName)];
  imwrite(F,savefile_f);
  
 
%%
%Rotating
  R=imrotate(I,10)
  R=imresize(R,[512 512]);
  savefile_r=['C:\Users\AugmentedImages\Rotate_',char(baseFileName)];
  imwrite(R,savefile_r);

  %Rotating
  R2=imrotate(I,-10)
  R2=imresize(R2,[512 512]);
  savefile_r2=['C:\Users\AugmentedImages\Rotate_',char(baseFileName)];
  imwrite(R2,savefile_r2);
  %%

%Shearing
a1 = -0.20;
T1 = maketform('affine', [1 0 0; a1 1 0; 0 0 1] );
orange = [0 0 0]';
R = makeresampler({'cubic','nearest'},'fill');
S = imtransform(I,T1,R,'FillValues',orange); 
S=imresize(S,[512 512]);
savefile_s=['C:\Users\AugmentedImages\Shear_',char(baseFileName)];
imwrite(S,savefile_s);

a2 = -0.20;
T2 = maketform('affine', [1 0 0; a2 1 0; 0 0 1] );
orange = [0 0 0]';
R = makeresampler({'cubic','nearest'},'fill');
S2 = imtransform(I,T2,R,'FillValues',orange); 
S2=imresize(S2,[512 512]);
savefile_s2=['C:\Users\AugmentedImages\Shear_',char(baseFileName)];
imwrite(S2,savefile_s2);

%%

%shifing

T3= maketform('affine', [1 0 0; 0 1 0; 10 -10 1]);   %# represents translation
Sh1 = imtransform(I, T3, ...
    'XData',[1 size(I,2)], 'YData',[1 size(I,1)]);

savefile_sh=['C:\Users\AugmentedImages\Shifting_',char(baseFileName)];
imwrite(Sh1,savefile_sh);


T4= maketform('affine', [1 0 0; 0 1 0; -10 10 1]);   %# represents translation
Sh2 = imtransform(I, T3, ...
    'XData',[1 size(I,2)], 'YData',[1 size(I,1)]);

savefile_sh2=['C:\Users\AugmentedImages\Shifting_',char(baseFileName)];
imwrite(Sh2,savefile_sh2);

end

