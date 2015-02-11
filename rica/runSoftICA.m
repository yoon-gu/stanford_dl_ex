%% We will use minFunc for this exercise, but you can use your
% own optimizer of choice
clear all;
close all; % only became important once the code started succeeding
addpath(genpath('../common/')) % path to minfunc
%% These parameters should give you sane results. We recommend experimenting
% with these values after you have a working solution.
global params;
DEBUG = false;
TEST_STL = false;
if DEBUG
    params.m = 100;
    params.patchWidth = 3;
    params.numFeatures = 5;
else    
    params.m=10000;%200000;% num patches
    params.patchWidth=9;% width of a patch
    params.numFeatures = 50; %50 % number of filter banks to learn
end
params.n=params.patchWidth^2; % dimensionality of input to RICA
params.lambda = 0.0005;%;%0.0005; % sparsity cost
params.epsilon = 1e-2;% epsilon to use in square-sqrt nonlinearity
zcaEpsilon = 1e-4; %%% originally 1e-4

% Load MNIST data set
data = loadMNISTImages('../common/train-images-idx3-ubyte');

% emulate parameters for STL exercise
if TEST_STL
    params.m = 200000;
    params.numFeatures = 32;
    data = sparse(data); % for 32-bit MATLAB - eat the overhead in Octave
    data = data(:, 1:round(size(data, 2) * (5/6)));
    data = full(data);
end

% random selection of patches for visualization
if isOctave(); 
    call_randi = @(imax, sz1, sz2) 1 + round((imax-1)*rand(sz1, sz2)); 
else
    rng('default'); % get numbers to match (most) of the figures
    call_randi = @randi; % otherwise it becomes undefined...stupid MATLAB...
end
randsel = call_randi(params.m,200,1); % A random selection of samples for visualization

%% Preprocessing
% Our strategy is as follows:
% 1) Sample random patches in the images
% 2) Apply standard ZCA transformation to the data
% 3) Normalize each patch to be between 0 and 1 with l2 normalization

% Step 1) Sample patches
patches = samplePatches(data,params.patchWidth,params.m);
figure('name', 'Raw patches'); display_network(patches(:, randsel));
if ~isOctave(); patches = single(patches); end

% Step 2) Apply ZCA
%for p=-6:3
%    x = zca2(patches, 10^p);
%    m = sqrt(sum(patches.^2) + (1e-8));
%    x = bsxfunwrap(@rdivide,patches,m);
%    figure('name', sprintf('ZCA-whitened with eps = 10^%d', p));
%    display_network(x(:, randsel));
%end
%return

patches = zca2(patches, zcaEpsilon);
if ~DEBUG; figure('name', 'ZCA-whitened patches'); display_network(patches(:, randsel)); end

% Step 3) Normalize each patch. Each patch should be normalized as     % PCA section: variance normalization is NOT needed??
% x / ||x||_2 where x is the vector representation of the patch 
% x = patches;
m = sqrt(sum(patches.^2) + (1e-8));
x = bsxfunwrap(@rdivide,patches,m);




%% Run the optimization
options.Method = 'lbfgs';
options.MaxFunEvals = Inf;
options.MaxIter = 500;
%options.display = 'off';
% options.outputFcn = @showBases;
if ~isOctave; options.useMex = false; end


% initialize with random weights
randTheta = randn(params.numFeatures,params.n)*0.01; % 1/sqrt(params.n);
randTheta = randTheta ./ repmat(sqrt(sum(randTheta.^2,2)), 1, size(randTheta,2));
randTheta = randTheta(:);

% check gradients numerically - from cnn/computeNumericalGradient.m
if DEBUG
    options.DerivativeCheck = 'off'; % not displaying results for regularization term??

    [unused_, grad] = softICACost(randTheta, x, params);    
    addpath ../cnn
    numGrad = computeNumericalGradient( @(theta) softICACost(theta, x, params), randTheta );
 
    disp([numGrad grad]);
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    disp(diff); 
    assert(diff < 1e-6,...
        'Difference too large. Check your gradient computation again');    
    disp 'Derivatives checked!'
    return
end


% optimize
tic;
[opttheta, cost, exitflag] = minFunc( @(theta) softICACost(theta, x, params), randTheta, options); % Use x or xw
fprintf('Optimization time (sec): %g\n', toc);

% display result
W = reshape(opttheta, params.numFeatures, params.n);
figure('name', 'Final filters')
display_network(W');
