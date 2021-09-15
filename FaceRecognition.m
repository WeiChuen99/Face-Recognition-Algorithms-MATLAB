function  outputLabel=FaceRecognition(trainPath, testPath)
%%   A simple face reconition method using cross-correlation based tmplate matching.
%    trainPath - directory that contains the given training face images
%    testPath  - directory that constains the test face images
%    outputLabel - predicted face label for all tested images 

%% Retrieve training images and labels
folderNames=ls(trainPath);
trainImgSet=zeros(600,600,3,length(folderNames)-2); % all images are 3 channels with size of 600x600
labelImgSet=folderNames(3:end,:); % the folder names are the labels
for i=3:length(folderNames)
    imgName=ls([trainPath, folderNames(i,:),'\*.jpg']);
    trainImgSet(:,:,:,i-2)= imread([trainPath, folderNames(i,:), '\', imgName]);
end

%% Prepare the training image: Here we simply use the gray-scale values as template matching. 
% You should implement your own feature extraction/description method here
trainTmpSet=zeros(600*600,size(trainImgSet,4)); % use 600x600 feature vector 
for i=1:size(trainImgSet,4)
    tmpI= rgb2gray(uint8(trainImgSet(:,:,:,i)));
    tmpI=double(tmpI(:))/255'; % normalise the intensity to 0-1& store the feature vector
    trainTmpSet(:,i)=(tmpI-mean(tmpI))/std(tmpI); % Use zero-mean normalisation. This is not neccessary if you use other gradient-based feature descriptor
end

%% Face recognition for the test images
testImgNames=ls([testPath,'*.jpg']);
outputLabel=[];
for i=1:size(testImgNames,1)
    testImg=imread([testPath, testImgNames(i,:)]);
    %perform the same pre-process as the train images
    tmpI= rgb2gray(uint8(testImg));
    tmpI=double(tmpI(:))/255';                % normalise the intensity to 0-1& store the feature vector
    tmpI=(tmpI-mean(tmpI))/std(tmpI);
    
    ccValue=trainTmpSet'*tmpI;                % perform dot product (cross correlationwith all the training images, and
    labelIndx=find(ccValue==max(ccValue));    % retrieve the label that correspondes to the largest value. 
    outputLabel=[outputLabel;labelImgSet(labelIndx(1),:)];   % store the outputLabels for each of the test image
end

