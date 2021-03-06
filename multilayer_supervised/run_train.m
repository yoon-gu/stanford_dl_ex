% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

%% setup environment
clear all; % prevents crashing on reloading 45 MB training data from disk

% experiment information
% a struct containing network layer sizes etc
ei = [];

% add common directory to your path for
% minfunc and mnist data helpers
addpath ../common;
addpath(genpath('../common/minFunc_2012/minFunc'));

%% load mnist data
[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();

%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% the architecture specified below should produce  100% training accuracy
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)

% dimension of input features
ei.input_dim = 784;

% number of output classes [fixed for digits task]
ei.output_dim = 10;

% sizes of all hidden layers and the output layer
ei.layer_sizes = [256, ei.output_dim]; % default
%ei.layer_sizes = [192, 64, ei.output_dim];

% scaling parameter for l2 weight regularization penalty
ei.lambda = 10;

% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function     % this was EASY relative to all the NN bookkeeping
ei.activation_fun = 'logistic'; % 'logistic', 'tanh', or 'rectified'

% toggle my paranoid error checking
ei.DEBUG = false; 
if ei.DEBUG
    % speed things up for debugging
    m = size(data_train, 2) / 1000;
    data_train = data_train(:, 1:m);
    labels_train = labels_train(1:m);
end

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% setup minfunc options
options = [];
options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';
options.useMex = true; % my additions
options.MaxIter = 250; % add this? it's running like hours without this option... my kingdom for a pickle
if ei.DEBUG; options.DerivativeCheck = 'on'; end

%% run training
tic;
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, data_train, labels_train);
fprintf('Optimization took %f seconds.\n', toc);

%% compute accuracy on the test and train set
[unused_, unused_, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
[unused_,pred] = max(pred);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f\n', acc_test);

[unused_, unused_, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[unused_,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f\n', acc_train);
