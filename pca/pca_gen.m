%%================================================================
%% Step 0a: Load data
%  Here we provide the code to load natural image data into x.
%  x will be a 784 * 600000 matrix, where the kth column x(:, k) corresponds to
%  the raw image data from the kth 12x12 image patch sampled.
%  You do not need to change the code below.

close all;
clear all;
addpath(genpath('../common'))
x = sparse(loadMNISTImages('../common/train-images-idx3-ubyte'));
[n m] = size(x);
figure('name','Raw images');
if isOctave(); 
    call_randi = @(imax, sz1, sz2) 1 + round((imax-1)*rand(sz1, sz2)); 
else
    rng('default'); % get numbers to match (most) of the figures
    call_randi = @randi; % otherwise it becomes undefined...stupid MATLAB...
end
randsel = call_randi(size(x,2),200,1); % A random selection of samples for visualization

% memory issues on 32-bit MATLAB
if ~isOctave(); 
    % trade time for storage...
    stride = 392; % hey, this works
    assert(mod(size(x, 1), stride) == 0)
    for i=1:stride:size(x, 1)
        block = i : i+stride-1;
        xSingle(block, :) = single(full(x(block, :)));
    end
    x = xSingle;
    clear xSingle;
end 

% raw data statistics
fprintf('max(x) = %g\n', max(max(x)))
fprintf('min(x) = %g\n', min(min(x)))
fprintf('mean(x) = %g\n', mean(mean(x)))
if isOctave(); fprintf('std(x) = %g\n', mean(std(x))); end

display_network(x(:,randsel));




%%================================================================
%% Step 0b: Zero-mean the data (by row)
%  You can make use of the mean and repmat/bsxfun functions.
    avg = mean(x, 1); % as per tutorial: "Mean pixel intensity for each patch"
    %x = x - repmat(avg, size(x, 1), 1); % their storage-hungry version
    x = bsxfun(@minus, x, avg);
    

%%================================================================
%% Step 1a: Implement PCA to obtain xRot
%  Implement PCA to obtain xRot, the matrix in which the data is expressed
%  with respect to the eigenbasis of sigma, which is the matrix U.
    Sigma = x * x' / size(x, 2);
    [U, S, V] = svd(Sigma); % uh, just copying tutorial now...
    xRot = U' * x; % "rotated version of the data
    %if ~isOctave(); clear x; end


%%================================================================
%% Step 1b: Check your implementation of PCA
%  The covariance matrix for the data expressed with respect to the basis U
%  should be a diagonal matrix with non-zero entries only along the main
%  diagonal. We will verify this here.
%  Write code to compute the covariance matrix, covar. 
%  When visualised as an image, you should see a straight line across the
%  diagonal (non-zero entries) against a blue background (zero entries).
    covar = xRot * xRot' / size(xRot, 2);
    if ~isOctave(); clear xRot; end
    
    fprintf('Norm of off-diagonal covariance matrix elements = %g (should be tiny)\n',...
        norm(covar - covar.*eye(size(covar))))

% Visualise the covariance matrix. You should see a line across the
% diagonal against a blue background.
figure('name','Visualisation of covariance matrix (should see solid or no line)');
imagesc(covar);



%%================================================================
%% Step 2: Find k, the number of components to retain
%  Write code to determine k, the number of components to retain in order
%  to retain at least 99% of the variance.
    k = min(find(cumsum(diag(S)) / sum(diag(S)) >= 0.99));
    fprintf('k = %d (n = %d)\n', k, size(S, 1))


%%================================================================
%% Step 3: Implement PCA with dimension reduction
%  Now that you have found k, you can reduce the dimension of the data by
%  discarding the remaining dimensions. In this way, you can represent the
%  data in k dimensions instead of the original 144, which will save you
%  computational time when running learning algorithms on the reduced
%  representation.
% 
%  Following the dimension reduction, invert the PCA transformation to produce 
%  the matrix xHat, the dimension-reduced data with respect to the original basis.
%  Visualise the data and compare it to the raw data. You will observe that
%  there is little loss due to throwing away the principal components that
%  correspond to dimensions with low variation.
%     Upca = U(:,1:k);
%     xHat = Upca * (Upca' * x);
%     xHat = U(:,1:k) * xRot(1:k, :); % too opaque
    xHat = U(:,1:k) * (U(:,1:k)' * x);


% Visualise the data, and compare it to the raw data
% You should observe that the raw and processed data are of comparable quality.
% For comparison, you may wish to generate a PCA reduced image which
% retains only 90% of the variance.

figure('name',['PCA processed images ',sprintf('(%d / %d dimensions)', k, n),'']);
display_network(xHat(:,randsel));
% figure('name','Raw images'); % meh, view earlier figure
% display_network(x(:,randsel));
if ~isOctave(); clear xHat; end % 32-bit MATLAB issues

%%================================================================
%% Step 4a: Implement PCA with whitening and regularisation
%  Implement PCA with whitening and regularisation to produce the matrix
%  xPCAWhite. 

epsilon = 1e-1; 
    % note that you do this on the ENTIRE xRot matrix, NOT just the k-subspace.
    % hence the need for "regularisation"?
    xPCAWhite = diag(1./sqrt(diag(S) + epsilon)) * U'*x;%xRot; % oh, they GAVE code in the "PCA" section.
    %xPCAWhite = bsxfun(@rdivide, U'*x, sqrt(diag(S) + epsilon));
    if ~isOctave(); clear x; end % 32-bit MATLAB issues

%% Step 4b: Check your implementation of PCA whitening 
%  Check your implementation of PCA whitening with and without regularisation. 
%  PCA whitening without regularisation results a covariance matrix 
%  that is equal to the identity matrix. PCA whitening with regularisation
%  results in a covariance matrix with diagonal entries starting close to 
%  1 and gradually becoming smaller. We will verify these properties here.
%  Write code to compute the covariance matrix, covar. 
%
%  Without regularisation (set epsilon to 0 or close to 0), 
%  when visualised as an image, you should see a red line across the
%  diagonal (one entries) against a blue background (zero entries).
%  With regularisation, you should see a red line that slowly turns
%  blue across the diagonal, corresponding to the one entries slowly
%  becoming smaller.
    covarPCA = xPCAWhite * xPCAWhite' / size(xPCAWhite, 2);


% Visualise the covariance matrix. You should see a red line across the
% diagonal against a blue background.
figure('name','Visualisation of covariance matrix (depends on epsilon in Step 4a)');
imagesc(covarPCA);


%%================================================================
%% Step 5: Implement ZCA whitening
%  Now implement ZCA whitening to produce the matrix xZCAWhite. 
%  Visualise the data and compare it to the raw data. You should observe
%  that whitening results in, among other things, enhanced edges.
    xZCAWhite = U * xPCAWhite;


% Visualise the data, and compare it to the raw data.
% You should observe that the whitened images have enhanced edges.
figure('name','ZCA whitened images');
display_network(xZCAWhite(:,randsel));
% figure('name','Raw images'); % meh, see earlier figure.
% display_network(x(:,randsel)); % saves memory for MATLAB
