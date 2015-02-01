function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolDim,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations
%activations = zeros(convDim,convDim,numFilters,numImages);

% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations
%activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);

%%% MY CODE HERE %%%
    % pre-pooled activations needed for backprop
    activations = cnnConvolve(filterDim, numFilters, images, Wc, bc);
    assert(isequal(size(activations), [convDim,convDim,numFilters,numImages]));

    % pooled (subsampled) features.
    activationsPooled = cnnPool(poolDim, activations);
    assert(isequal(size(activationsPooled), [outputDim,outputDim,numFilters,numImages]));

% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
activationsPooled = reshape(activationsPooled,[],numImages);
%activations = reshape(activations, [], numImages); % this too, for backprop?

%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.
%probs = zeros(numClasses,numImages);

%%% MY CODE HERE %%%
    % from supervised_dnn_cost.m
    probs = calc_softmax_probabilities(bsxfun(@plus, Wd*activationsPooled, bd));
    assert(isequal(size(probs), [numClasses,numImages]));

%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

%cost = 0; % save objective into cost

%%% MY CODE HERE %%%
    cost = calc_cost(probs, labels);

% Makes predictions given probs and returns without backproagating errors.
if pred
    [unused_,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.

%%% MY CODE HERE %%%
    dd = calc_output_delta(probs, labels);
    
    dcSubsampled = Wd' * dd; 
    dcSubsampled = reshape(dcSubsampled, [outputDim,outputDim,numFilters,numImages]);
    
    for filterNum = 1:numFilters
        for imageNum = 1:numImages
            a = activations(:,:,filterNum,imageNum);
            dc(:,:,filterNum, imageNum) = ... % upsample using kron(), as per instructions
                kron(dcSubsampled(:,:,filterNum,imageNum), ones(poolDim)) / poolDim^2 ...
                .* a .* (1-a) ... % f' for sigmoid nonlinearity
            ;
        end
    end
    
        %size(Wd) % [10 2000]
        %size(bd)
        %size(bc) % [20 1] - one per Wc matrix.
        %size(Wc) % [9 9 20]; agrees with spec
        %size(dd) % [ 10 256]
        %size(activations) % [8000 256]
        %size(activationsPooled) % [2000 256]. m = 2000?
        %size(activations) % [convDim,convDim,numFilters,numImages] % need to unroll? oh boy...
    



%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

%%% YOUR CODE HERE %%%

%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

end




function pred_prob = calc_softmax_probabilities(Z)  
    % input: Z(1:output_classes, 1:num_examples)
    % output: pred_prob(1:output_classes, 1:num_examples)
    % uh, copy/pasted from supervised_dnn_cost.m...
        % too short to merit its own file...
        % it was largely based on ex1c_softmax in the first place
    p_unnormalized = exp(Z);
    pred_prob = bsxfun(@rdivide, p_unnormalized, sum(p_unnormalized, 1));    
end



function cost = calc_cost(P, y)
    % copy/pasted from supervised_dnn_cost.m...
    logP = log(P);   
    cost = -mean(logP( observed(logP, y) )); 
end



function obs = observed(P, y)
    % inputs
    %   M = (n x m) matrix for m examples
    %   y = (m x 1) vector. each entry is the observed row of the mth example
    % output
    %   obs = (n x m) matrix with ones at observations.
    
    % original implementation from supervised_dnn_cost.m
    %m = size(M, 2);
    %assert(isequal(size(y), [m 1]));
    %obs = sub2ind(size(M), y', 1:m);
    
    % the prescribed implementation for this exercise, using sparse()
    assert(size(y, 2) == 1);
    obs = find(sparse(y', 1:numel(y), ones(size(y'))) ~= 0);
    
end



function delta = calc_output_delta(P, y)
    % copy/pasted from supervised_dnn_cost.m...
    delta = P;    
    obs = observed(P, y);
    delta(obs) = delta(obs) - 1;
end