function[seg_im] = sobel_segmentation(image_file_name)

%read image
image = image_file_name;

% 1) turn into binary by threshold
threshold_level = graythresh(image); %threshold of image
binaryFudgeFactor = 0.7;
binarize = imbinarize(image,threshold_level*binaryFudgeFactor);

% 2) complement of the binary image
image_complement = imcomplement(binarize); 

% 3) Detect entire cell using Sobel
[~, threshold] = edge(image_complement,'sobel');
fudgeFactorDetection = .5;
sobelDetection = edge(image_complement, 'sobel', threshold*fudgeFactorDetection);

% 4) Dilate the Image
se90 = strel('line', 3, 90);
se0 = strel('line',3,0); 
imageDilate = imdilate(sobelDetection, [se90, se0]);

% 5) Fill Interior Gaps
imageFill = imfill(imageDilate, 'holes');

% 6) Remove Connected Objects on Border
imageNoborder = imclearborder(imageFill, 4);

% 7) Smoothen the Object
seD = strel('diamond',2);
imageErode = imerode(imageNoborder,seD);
imageErode = imerode(imageErode,seD);

% 8) removes objects smaller than minPixels
minPixels = 150;
imageFinal = bwareaopen(imageErode, minPixels); 

%show the image results
%smontage({image, image_complement, sobelDetection, imageDilate, imageFill, imageNoborder, imageFinal},'Size',[2 4])

seg_im = logical(BWfinal_area);
end
