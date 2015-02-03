function features = feedfowardRICA(filterDim, poolDim, numFilters, images, W)
% feedfowardRICA Returns the convolution of the features given by W with
% the given images. It should be very similar to cnnConvolve.m+cnnPool.m 
% in the CNN exercise, except that there is no bias term b, and the pooling
% is RICA-style square-square-root pooling instead of average pooling.
%
% Parameters:
%  filterDim - filter (feature) dimension
%  numFilters - number of feature maps
%  images - large images to convolve with, matrix in the form
%           images(r, c, image number)
%  W    - W should be the weights learnt using RICA
%         W is of shape (filterDim,filterDim,numFilters)
%
% Returns:
%  features - matrix of convolved and pooled features in the form
%                      features(imageRow, imageCol, featureNum, imageNum)
global params;
numImages = size(images, 3);
imageDim = size(images, 1);
convDim = imageDim - filterDim + 1;

features = zeros(convDim / poolDim, ...
        convDim / poolDim, numFilters, numImages);
poolMat = ones(poolDim);
% Instructions:
%   Convolve every filter with every image just like what you did in
%   cnnConvolve.m to get a response.
%   Then perform square-square-root pooling on the response with 3 steps:
%      1. Square every element in the response
%      2. Sum everything in each pooling region
%      3. add params.epsilon to every element before taking element-wise square-root
%      (Hint: use poolMat similarly as in cnnPool.m)
    % uh, wouldn't it have been better to call cnnConvolve() instead?? sigh.


for imageNum = 1:numImages
  if mod(imageNum,500)==0 || params.DEBUG
    fprintf('forward-prop image %d / %d\n', imageNum, numImages);
    if isOctave(); fflush(stdout); end
  end
  for filterNum = 1:numFilters

    %filt = zeros(8,8); % You should replace this                           % this was the wrong size?
    % Form W, obtain the feature (filterDim x filterDim) needed during the
    % convolution
    %%% MY CODE HERE %%% - copy/pasted from cnnConvolve.m....
        convolvedImage = zeros(convDim, convDim); % useful for matrix size checking, I suppose
        filt = W(:,:,filterNum);
        assert(isequal(size(filt), [filterDim filterDim]));

    % Flip the feature matrix because of the definition of convolution, as explained later
    filt = rot90(squeeze(filt),2);
      
    % Obtain the image
    im = squeeze(images(:, :, imageNum));

    %resp = zeros(convDim, convDim); % You should replace this
    % Convolve "filter" with "im" to find "resp"
    % be sure to do a 'valid' convolution
    %%% MY CODE HERE %%% - copy/pasted from cnnConvolve.m....
        convolvedImage = convolvedImage + conv2(im, filt, 'valid'); % 'no bias term' - as per spec above
        resp = sigmoid(convolvedImage); % hmmm... i guess so...
        assert(isequal(size(resp), [convDim convDim]))
    
    % Then, apply square-square-root pooling on "resp" to get the hidden
    % activation "act"
    %act = zeros(convDim / poolDim, convDim / poolDim); % You should replace this
    %%% MY CODE HERE %%%
        convolved = conv2(resp.^2, ones(poolDim, poolDim), 'valid');
        subsampled = convolved(1:poolDim:convDim, 1:poolDim:convDim);
        act = sqrt(subsampled + params.epsilon); % no need to divide by pooDim^2?
        assert(isequal(size(act), [convDim / poolDim, convDim / poolDim]));
        assert(mod(convDim, poolDim) == 0);
    
    features(:, :, filterNum, imageNum) = act;
  end
end


end

