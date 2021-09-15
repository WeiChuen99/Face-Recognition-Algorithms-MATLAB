% trainPath='.\FaceDatabase\Train\'; % provide full path here
% testPath='.\FaceDatabase\Test\';
function  outputLabel=FaceRecognition2(trainPath, testPath)

%%  Load Train and test Images.
folderNames=ls(trainPath);
trainImgSet=zeros(600,600,3,length(folderNames)-2); % all images are 3 channels with size of 600x600
labelImgSet=folderNames(3:end,:); % the folder names are the labels
for i=3:length(folderNames)
    imgName=ls([trainPath, folderNames(i,:),'\*.jpg']);
    trainImgSet(:,:,:,i-2)= imread([trainPath, folderNames(i,:), '\', imgName]);
end
trainImgSz = size(trainImgSet);

testImgNames=ls([testPath,'*.jpg']);
outputLabel=[];
testImgSet = zeros(600,600,3,length(testImgNames));
for i=1:size(testImgNames,1)
    testImgSet(:,:,:,i) = imread([testPath, testImgNames(i,:)]);
end
testImgSz = size(testImgSet);

% viola jones crop train images
faceDetectorfront = vision.CascadeObjectDetector('FrontalFaceLBP');

scale = 224;
trainCropSet = CropImg(trainImgSet,faceDetectorfront,scale);

testCropSet = CropImg(testImgSet,faceDetectorfront,scale);
%% train and test surf feature extraction
strongestPoints = 10; 
trainfeatures = extract_surf(trainImgSz, trainCropSet, strongestPoints);

testfeatures = extract_surf(testImgSz, testCropSet,strongestPoints);

outputLabel = [];
for i = 1:testImgSz(4)
    if mod(i, 100) == 0
%         disp(["test ",i]);
    end
    distance2 = zeros(trainImgSz(4),1); 
    distance2= distance2 + 4;
    for j = 1:trainImgSz(4)
        [~,dist] = matchFeatures(testfeatures(:,:,i),trainfeatures(:,:,j),...
            'MatchThreshold' ,1);
        if isempty(dist)
            continue
        else
           distance2(j,1) = sum(dist)/length(dist);
        end
    end
    
    labelIndx=find(distance2==min(distance2)); 
    outputLabel=[outputLabel;labelImgSet(labelIndx(1),:)];
end

% load testLabel
% correctP=0;
% for i=1:size(testLabel,1)
%    if strcmp(outputLabel(i,:),testLabel(i,:))
%        correctP=correctP+1;
%    end
% end
% %
% recAccuracy=correctP/size(testLabel,1)*100;  %Recognition accura
% disp([recAccuracy, "%"])

end
%%
function trainCropSet = CropImg(ImgSet,faceDetectorfront, scale)
    trainCropSet = cell(size(ImgSet,4),1);
    for i = 1:size(ImgSet,4)
        im = rgb2gray(uint8(ImgSet(:,:,:,i)));
        bboxes = faceDetectorfront(im);

        if isempty(bboxes)
            bboxes = [0 0 size(ImgSet,1) size(ImgSet,2)];
        end
        % cropped = imcrop(im, [bboxes(1)-50,bboxes(2)-50,bboxes(3)+50,bboxes(4)+50]);
       
        cropped = imcrop(im, bboxes(1,:));

        %resize image according to scale factor
        try
            scaleFactor = scale/size(cropped,1);
        catch
            
        end
        cropped = imresize(cropped, scaleFactor);
        trainCropSet{i} = cropped;
    end
end

function trainfeatures = extract_surf(ImgSz, CropSet,strongestPoints)
    nstrongestpoint  = strongestPoints;
%     trainfeatures = cell(size(trainImgSet,4),2);

    trainfeatures = zeros(nstrongestpoint,64,ImgSz(4));
    for i=1:ImgSz(4)
        im = CropSet{i};
        points = detectSURFFeatures(im);
        points = points.selectStrongest(nstrongestpoint);
        [features, validpoints] = extractFeatures(im,points, 'Method', 'SURF');
    %     if isempty(features)
%         trainfeatures{i,1} = features;
%         trainfeatures{i,2} = validpoints;

        if isempty(features)
            trainfeatures(:,:,i) = zeros(nstrongestpoint,64);
        elseif size(features,1) < nstrongestpoint
            trainfeatures(:,:,i) = padarray(features,nstrongestpoint - size(features,1) ,0,'post');
        else
            trainfeatures(:,:,i) = features;
        end
    end
end