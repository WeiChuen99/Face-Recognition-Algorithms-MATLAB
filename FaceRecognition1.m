function  outputLabel=FaceRecognition1(trainPath, testPath)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%please download model from model 1 from onedrive
%link = https://uniofnottm-my.sharepoint.com/:f:/g/personal/hcyws1_nottingham_ac_uk/EvbdbKs5adBAnpb-_9O2mI8BD_YCurggG7p_xhEl2KHfLQ?e=ffefbs
%please place folder in current directory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Retrieve training images and labels

imdsTrain = imageDatastore(trainPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsTest = imageDatastore(testPath, 'IncludeSubfolders', true);

imdsTrain.ReadFcn = @customreader1;
imdsTest.ReadFcn = @customreader1;

%% Load network
protofile = '.\vgg_face_caffe\VGG_FACE_deploy.prototxt';
datafile = '.\vgg_face_caffe\VGG_FACE.caffemodel';
net = importCaffeNetwork(protofile,datafile);

%replace output layers of pre trained model
layers = net.Layers;
layers(39) = fullyConnectedLayer(numel(categories(imdsTrain.Labels)), 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
layers(41) = classificationLayer('Name','classoutput', 'Classes', categories(imdsTrain.Labels));

%freeze weights of earlier layers
layers(1:36) = freezeWeights(layers(1:36));
%% training
tic;
initlearnRate = 0.0005;
maxepochs = 10;
solver = 'sgdm';
minibatch = 25;
options = trainingOptions(solver, ...
    'MaxEpochs',maxepochs,...
    'InitialLearnRate',initlearnRate, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'MiniBatchSize', minibatch);

convnet = trainNetwork(imdsTrain,layers,options); % Make a new network and keep training
trainTime = toc;

ypred = classify(convnet,imdsTest);

outputLabel = char(ypred(:,:));