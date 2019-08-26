%passed an image file that we already read
function[seg_im] = canny_segmentation(I)
 
I2 = imadjust(I); %add contrast to the image
B = imgaussfilt(I, 2.5); %gauss filter with sigma 2.5
 
fudgeFactor = 1.6; %add change to the threshold in canny edge detection
[~, threshold] = edge(B,'canny'); %detects edges in image I using the edge-detection algorithm specified by method.
BR_BWs = edge(B, 'canny', threshold*fudgeFactor, 0.25); %returns all edges that are stronger than threshold. specifies sigma
 
%Dilate the Image
se90 = strel('line',4, 90);
se0 = strel('line',3,0);
BR_BWsdil = imdilate(BR_BWs, [se90, se0]); 
 
%Fill Interior Gaps
BWdfill = imfill(BR_BWsdil, 'holes');

%eroding - removing extra lines
SE = strel('line',7, 90); %structuring element
out_2 = imerode(BWdfill,SE);
SE = strel('line',5, 0); %structuring element
out_2 = imerode(out_2,SE);
 
out_2 = bwareaopen(out_2, 160); %remove extra spots less than 160 pixels
 
seg_im = out_2; %return segmented image
 
end
