function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)
    % gah? ceCost, wCost return values are defunct...

%debug_on_error(1, 'local'); % ugh, couldn't really get octave debugger to work... isn't that what matlab is for??
    
%% default values
po = false;
if exist('pred_only','var') % uh, does matlab automatically do optional arguments?
  po = pred_only;
end;
cost = 0;
grad = zeros(size(theta));
pred_prob = zeros(ei.output_dim, size(data, 2)); % THIS is the correct "empty stub" default value

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);    % activations. last layer is stored as pred_prob instead
gradStack = params2stack(grad, ei);

nl = numHidden+2;
Z = cell(nl, 1);
delta = cell(nl, 1);



%% forward prop
    % input layer - to avoid special code JUST for the first iteration
    hAct{1} = data;
    
    % propagate to hidden layers - just reading off the tutorial. easy peasy!
    for l=2:nl
        
        % z(l) = W(l-1)x + b(l-1) from definition of the linear signal
        %Z{l} = (stack{l-1}.W)*hAct{l-1}; + repmat(stack{l-1}.b, 1, size(hAct{l-1}, 2)); % which is less opaque? i'm not sure...
        Z{l} = bsxfun(@plus, (stack{l-1}.W)*hAct{l-1}, stack{l-1}.b); % but this SHOULD require less storage?
        
        % a(l) = f(z(l)) by definition of nonlinear activations for hidden layers.
        % as per instructions, output layer will be treated separately with softmax/maxent/Boltzmann in cost code
        if (l < nl)
            hAct{l} = f(Z{l}, ei);
        end        
    end
    
    % final layer outputs softmax (maxent) class probabilities
        % COULD in principle exploit monotonicity of exp(.) to make predictions, but the user spec is to output a probability...
    pred_prob = calc_softmax_probabilities(Z{nl});
    
    
%% return here if only predictions desired.
    if po
      cost = -1; %ceCost = -1; wCost = -1; numCorrect = -1;
      grad = [];  
      return;
    end;

    
%% compute cost
    % cost and initial gradient can just use softmax_regression_vec, right? 
        % no, gradient is different. plus, we already have z = inner products
    % wait, theta values are the last weight layer?? so unclear... but that's what civilstat does.
    assert(size(Z{nl}, 1) == ei.output_dim);
    
    % system-specific parts kept in subroutines to avoid obfuscating basic backprop logic
    cost = calc_cost(pred_prob, labels);
    

%% compute gradients using backpropagation
    delta{nl} = calc_output_delta(pred_prob, labels);


%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%


%% Some error checking
    assert(isempty(Z{1}), 'The first member of Z was supposed to be thrown away to normalize numbering');
    
    % delta{1} also empty?


%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



function a = f(Z, ei)
    % hidden unit nonlinearity
    switch ei.activation_fun
        case 'logistic'
            % from ex1/sigmoid.m
            a = 1 ./ (1+exp(-Z));
        otherwise
            message = sprintf('Unknown activation function: %s\n', ei.activation_fun);
            assert(false, message);
    end
end



function pred_prob = calc_softmax_probabilities(Z)
    % computes predicted class probabilities from softmax function on final hidden layer signals Z
    
    % reusing lots of code from softmax_regression_vec.m...
    n = size(Z, 1); % rows = classes
    m = size(Z, 2); % cols = data examples 
    assert(n == 10, '10 classes for digits data');
    
    % CANNOT fix weights for last class to 0 ("zero energy level")! not without changing initialize_weights()... meh
    p_unnormalized = exp(Z);
    pred_prob = bsxfun(@rdivide, p_unnormalized, sum(p_unnormalized, 1));    
end



function cost = calc_cost(P, y)
    assert(size(P, 1) == 10, 'Not using 10-digit data??');

    % calculates cost J(theta)
    logP = log(P);   
    
    % the slick "indicator function" from softmax_regression_vec.m
    % annoyingly, y = labels is a COLUMN vector here, but it was a ROW vector in softmax_regression_vec
    cost = -sum(logP( observed(logP, y) )); 
end



function obs = observed(M, y)
    % inputs
    %   M = (n x m) matrix for m examples
    %   y = (m x 1) vector. each entry is the observed row of the mth example
    % output
    %   obs = (1 x m) vector = indices of the observed entries in M
    m = size(M, 2);
    assert(isequal(size(y), [m 1]));
    obs = sub2ind(size(M), y', 1:m);
end



function delta = calc_output_delta(P, y)
    % SIMILAR to softmax_regression_vec.m, but there is no multiplication by X or summation over examples.
    delta = P;
    
    obs = observed(P, y);
    delta(obs) = delta(obs) - 1; % inspired by civilstat
end