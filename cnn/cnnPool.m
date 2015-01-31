function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.

    % """
    % This can be done
    % efficiently using the conv2 function as well. The inputs are the responses of each    
    % image with each filter computed in the previous step. Convolve each of these with a
    % matrix of ones followed by a subsampling and averaging. Make sure to use the “valid”
    % border handling convolution.
    % """
    % this method sounds wasteful...doesn't subsampling throw away a bunch of work?
    % oh but the alternative is a loop over pooling coordinates, which is even slower??
        % but wait, if you can do the subsampling AFTER, couldn't you do it before too??
        % i guess it's a single line to subsample, but a loop to get blocks?
    %assert(mod(convolvedDim, poolDim) == 0); % meh, implicit in pooledFeatures allocation above
    for imageNum = 1:numImages
        for filterNum = 1:numFilters        
            features = convolvedFeatures(:,:,filterNum,imageNum);
            convolved = conv2(features, ones(poolDim, poolDim), 'valid');
            subsampled = convolved(1:poolDim:convolvedDim, 1:poolDim:convolvedDim); % wasteful! you waste!
            pooledFeatures(:,:, filterNum, imageNum) = subsampled / (poolDim^2);
        end
    end
    
    % alternative approach : meh, would double storage requirements
        % get vector of blocks. double loop, or...?
        % apply mean(mean()) to each block
        % reshape to matrix
    
    
    % http://stackoverflow.com/questions/25964034/how-do-i-average-columns-block-wise
        % reshape doesn't work, since it goes in column order...
    
    % http://www.mathworks.com/matlabcentral/newsreader/view_thread/17463
    % cleaner code, but turns out to be SLOWER than conv2. colfilt() is also slow?
    %for imageNum = 1:numImages
    %    for filterNum = 1:numFilters
    %        pooledFeatures(:,:, filterNum, imageNum) = blkproc( ...
    %            convolvedFeatures(:,:,filterNum,imageNum), ...
    %            [poolDim poolDim], ...
    %            @(m) mean(mean(m)) ...
    %        );
    %    end
    %end

end

