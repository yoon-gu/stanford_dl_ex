%% CS294A/CS294W Self-taught Learning Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  self-taught learning. You will need to complete code in feedForwardAutoencoder.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises.
%

    % is it just me, or do the minor hacks that let this run on 32-bit MATLAB make it a LOT faster??
    % like, the ENTIRE script runs in ~2 min, instead of 20+?

clear all; 
close all;
tstart = tic;


%% ======================================================================
%  STEP 0: Here we provide the relevant parameters values that will
%  allow your RICA to get good filters; you do not need to 
%  change the parameters below.
addpath(genpath('..')) % apparently this adds all subdirectories too (should've noticed from minFunc)
imgSize = 28;
global params;
params.patchWidth=9;           % width of a patch
params.n=params.patchWidth^2;   % dimensionality of input to RICA
params.lambda = 0.0005;   % sparsity cost
params.numFeatures = 32; % number of filter banks to learn
params.epsilon = 1e-2;   
params.DEBUG = false;

if ~isOctave(); options.useMex = false; end % Octave and MATLAB mex files can't commingle...can they?

DEBUG = params.DEBUG;
if DEBUG
    numPatches = 2000;
    if isOctave()
        fractionUnlabeled = 0.99; % blow through supervised stage to test syntax
    else
        fractionUnlabeled = 5/6; % 32-bit MATLAB has trouble storing 99% of the training data (10000-image test set is ok)
    end
else
    numPatches = 200000; % 200000 for production; 20000 can run in ~2.5 min
    fractionUnlabeled = 5/6;
end



%% ======================================================================
%  STEP 1: Load data from the MNIST database
%
%  This loads our training and test data from the MNIST database files.
%  We have sorted the data for you in this so that you will not have to
%  change it.

% Load MNIST database files
mnistData   = sparse(loadMNISTImages('../common/train-images-idx3-ubyte'));         % JUST enough to get 32-bit MATLAB to run
mnistLabels = loadMNISTLabels('../common/train-labels-idx1-ubyte');

numExamples = size(mnistData, 2);
% most of the data are pretended to be unlabelled
numUnlabeled = round(numExamples*fractionUnlabeled);
unlabeledSet = 1:numUnlabeled; 
unlabeledData = full(mnistData(:, unlabeledSet));


% the rest are equally splitted into labelled train and test data
numLabeled = numExamples - numUnlabeled;
trainSet = (numUnlabeled + 1) : (numUnlabeled + round(numLabeled/2)); 
testSet = (numUnlabeled + round(numLabeled/2) + 1) : (numExamples); 
trainData   = full(mnistData(:, trainSet));
trainLabels = mnistLabels(trainSet)' + 1; % Shift Labels to the Range 1-10
% only keep digits 0-4, so that unlabelled dataset has different distribution       % self-taught, not semi-supervised
% than the labelled one.
removeSet = find(trainLabels > 5);
trainData(:,removeSet)= [] ;
trainLabels(removeSet) = [];

testData   = full(mnistData(:, testSet));
testLabels = mnistLabels(testSet)' + 1;   % Shift Labels to the Range 1-10
% only keep digits 0-4
removeSet = find(testLabels > 5);
testData(:,removeSet)= [] ;
testLabels(removeSet) = [];

if ~isOctave(); clear mnistData; end % every little bit helps?


% Output Some Statistics
fprintf('# examples in unlabeled training set: %d\n', size(unlabeledData, 2));
fprintf('# examples in supervised training set: %d\n', size(trainData, 2));
fprintf('# examples in supervised testing set: %d\n\n', size(testData, 2));
if isOctave(); fflush(stdout); end



%% ======================================================================
%  STEP 2: Train the RICA
%  This trains the RICA on the unlabeled training images. 

%  Randomly initialize the parameters
randTheta = randn(params.numFeatures,params.n)*0.01;  % 1/sqrt(params.n);
randTheta = randTheta ./ repmat(sqrt(sum(randTheta.^2,2)), 1, size(randTheta,2)); 
randTheta = randTheta(:);

% subsample random patches from the unlabelled+training data
if isOctave()
    patches = samplePatches([unlabeledData,trainData],params.patchWidth,numPatches);
else
    patches = samplePatches(unlabeledData, params.patchWidth, numPatches); % horzcat JUST too big for 32-bit MATLAB
end

%configure minFunc
options.Method = 'lbfgs';
options.MaxFunEvals = Inf;
options.MaxIter = 1000;
if DEBUG; options.MaxIter = 10; end
options.outputFcn = @showBases;
% You'll need to replace this line with RICA training code
%opttheta = randTheta;
%V = eye(params.n);

%  Find opttheta by running the RICA on all the training patches.
%  You will need to whitened the patches with the zca2 function 
%  then call minFunc with the softICACost function as seen in the RICA exercise.
    %%% MY CODE HERE %%% - copy/pasted from runSoftICA.m....(too short...)
    % Apply ZCA. pca_gen.m: epsilon = 1e-1; zca2.m: epsilon = 1e-4...
    [x, U, S, V] = zca2(patches, 0.05); % orthonormal ICA requires epsilon = 0, but RICA doesn't??
    
    
    % I actually think the data look BLURRIER after ZCA(1e-4)

    % sanity check: visualize patches before/after ZCA whitening
        % yep, it's NOT my imagination. zca2(epsilon = 0.1) DOES seem to perform better!
            % filtered vs raw patches don't look like blurred noise (like they do for 1e-4)
            % "the long tail" (old tutorial "Data Preprocessing) starts around eigenvalues <= ~0.06
    if isOctave() % 'learned filters' updates overwrite these in MATLAB
        patchesToVisualize = 1:params.numFeatures;
        figure('name', 'Raw patches');
        display_network(patches(:, patchesToVisualize));
    end
    %assert(false, 'Learned filters overwrite patches...') % matlab only?

    %assert(false, 'here to analyze S')
    
    if ~isOctave(); clear patches; end
        % zca2(patches, 10) gave 98% test accuracy and 100% training accuracy... just a lucky fluctuation?
    
    % Normalize each patch - should i not do this?? still don't really understand RICA...
    m = sqrt(sum(x.^2) + (1e-8)); 
    x = bsxfunwrap(@rdivide,x,m);

    if isOctave()
        figure('name', 'ZCA-whitened patches');
        display_network(x(:, patchesToVisualize));
    end
    figure('name', 'Learned filters');
    
    % optimize (train RICA)
    tic;
    opttheta = minFunc( @(theta) softICACost(theta, x, params), randTheta, options );
    fprintf('RICA unsupervised training time (sec): %g\n', toc);
    if isOctave(); fflush(stdout); end

% reshape visualize weights
W = reshape(opttheta, params.numFeatures, params.n);
display_network(W');

    % TODO: mean-normalize supervised data with RICA mean?


%% ======================================================================
%% STEP 3: Extract Features from the Supervised Dataset
% pre-multiply the weights with whitening matrix, equivalent to whitening       % POST-multiply the weights?
% each image patch before applying convolution. V should be the same V
% returned by the zca2 when you whiten the patches.
W = W*V;
%  reshape RICA weights to be convolutional weights.
W = reshape(W, params.numFeatures, params.patchWidth, params.patchWidth);
W = permute(W, [2,3,1]);

%  setting up convolutional feed-forward. You do need to modify this code.      % do NOT need to modify?
filterDim = params.patchWidth;
poolDim = 5;                                                                    % this should really be with its brethren in params
numFilters = params.numFeatures;
trainImages=reshape(trainData, imgSize, imgSize, size(trainData, 2));
testImages=reshape(testData, imgSize, imgSize, size(testData, 2));

%  Compute convolutional responses
%  Completed feedfowardRICA.m.
tic;
trainAct = feedfowardRICA(filterDim, poolDim, numFilters, trainImages, W);
testAct = feedfowardRICA(filterDim, poolDim, numFilters, testImages, W);
fprintf('RICA feature extraction (convolution) time (sec): %g\n', toc);

%  reshape the responses into feature vectors
featureSize = size(trainAct,1)*size(trainAct,2)*size(trainAct,3);
trainFeatures = reshape(trainAct, featureSize, size(trainData, 2));
testFeatures = reshape(testAct, featureSize, size(testData, 2));


    
%% ======================================================================
%% STEP 4: Train the softmax classifier

numClasses  = 5; % doing 5-class digit recognition
% initialize softmax weights randomly
randTheta2 = randn(numClasses, featureSize)*0.01;  % 1/sqrt(params.n);
randTheta2 = randTheta2 ./ repmat(sqrt(sum(randTheta2.^2,2)), 1, size(randTheta2,2)); 
    % need a bit of pre/post processing to handle softmax_regression_vec's 
    % non-standard quirk of fixing theta = 0 for last class
    randTheta2 = randTheta2(1:numClasses-1, :);
randTheta2 = randTheta2';
randTheta2 = randTheta2(:);

%  Use minFunc and softmax_regression_vec from the previous exercise to 
%  train a multi-class classifier. 
options.Method = 'lbfgs';
options.MaxFunEvals = Inf;
options.MaxIter = 300;
if isfield(options, 'outputFcn'); options = rmfield(options, 'outputFcn'); end

% optimize
%%% MY CODE HERE %%% - MODIFIED from ex1c_softmax.m
    tic;
    opttheta2 = [ ...
        reshape(...
            minFunc(@softmax_regression_vec, randTheta2(:), options, trainFeatures, trainLabels), ...
            featureSize, numClasses-1 ...
        ), ...
        zeros(featureSize, 1) ...
    ];
    fprintf('Softmax classifier training time (sec): %g\n', toc);
    



%%======================================================================
%% STEP 5: Testing 
% Compute Predictions on tran and test sets using softmaxPredict
% and softmaxModel
%%% MY CODE HERE %%%
    % uh, no need to normalize, since exp(.) is monotonic
    [unused_, train_pred] = max(opttheta2'*trainFeatures, [], 1);
    [unused_, pred] = max(opttheta2'*testFeatures, [], 1);
    
% Classification Score
fprintf('Train Accuracy: %f%%\n', 100*mean(train_pred(:) == trainLabels(:)));
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));
% You should get 100% train accuracy and ~99% test accuracy. With random
% convolutional weights we get 97.5% test accuracy. Actual results may
% vary as a result of random initializations

fprintf('Total runtime (sec): %g\n', toc(tstart));

