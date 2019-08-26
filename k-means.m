%pass in image
function[seg_im] = kmeans_segmentation(I) 
 
I = imgaussfilt(I, 2.5); %gaussian filter with sigma 2.5
 
[L,~] = imsegkmeans(I,2); %L and centers are returned, 2 clusters
B = labeloverlay(I,L);
 
%Supplement image with information about the texture in the neighborhood of each pixel. 
%To obtain the texture information, filter a grayscale version of the image with a set of Gabor filters.
%Create a set of 24 Gabor filters, covering 6 wavelengths and 4 orientations.
wavelength = 2.^(0:5) * 3;
orientation = 0:45:135;
g = gabor(wavelength,orientation);
 
%Convert the image to grayscale.
I = rgb2gray(im2single(B));
 
gabormag = imgaborfilt(I,g);
 
for i = 1:length(g)
    sigma = 0.5*g(i).Wavelength;
    gabormag(:,:,i) = imgaussfilt(gabormag(:,:,i),3*sigma); %smooth gabor info
end

nrows = size(I,1);
ncols = size(I,2);
[X,Y] = meshgrid(1:ncols,1:nrows);
 
featureSet = cat(3,I,gabormag,X,Y);
 
L2 = imsegkmeans(featureSet,2,'NormalizeInput',true); %do kmeans on new featureset
C = labeloverlay(I,L2);
 

%histogram method to get the smallest percentage of pixels in the image. This percentage is the cluster information that is stuck in 3 layers of this image.
%essentially get the cluster that should be the mitochondria!! Here I choose the 2nd layer since it generally has the clustering that contains the mitochondria. Sometimes it doesn’t and it returns blank or extra spots we don’t want.
 
[N, ~, bin] = histcounts(C(:,:,2), 4); %number of bins we want is 3 for 3 clusters
%N is the number of items in each bin (first output of histcounts)
 
[M, Index] = min(N); %M is smallest element. So Index is the bin number
 
[width, height, ~] = size(C); %i think height was a 2d matrix on accident? 
  
Final = zeros(width,height); %create the final image
 
for col=1:height
    for row=1:width
        if bin(row, col) == Index(1)
            Final(row, col) = 1; %put the pixels in where they should be white
        end
    end
end
 
Final3 = bwareaopen(Final, 100); #remove extra spots that are less than 100 pixels.
  
seg_im = Final3; %return final image
 
end
