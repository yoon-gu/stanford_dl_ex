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
pred_prob = 0;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+2, 1);    % activations? throw away first cell to make numbering match.
gradStack = params2stack(grad, ei);

Z = cell(numel(hAct), 1);
nl = numHidden+2;

%% forward prop - just reading off the tutorial. easy!
hAct{1} = data;
for l=1:nl-1
    
    % z(l+1) = W(l)x + b(l)
    %Z{l+1} = (stack{l}.W)*hAct{l}; + repmat(stack{l}.b, 1, size(hAct{l}, 2)); % which is less opaque? i'm not sure...
    Z{l+1} = bsxfun(@plus, (stack{l}.W)*hAct{l}, stack{l}.b); % but this SHOULD require less storage?
    
    % a(l+1) = f(z(l+1)) for hidden layers.
    % as per instructions, output layer will be treated separately with softmax/maxent/Boltzmann in cost code
    if (l+1 < nl)
        hAct{l+1} = f(Z{l+1}, ei);
    end
end

%% return here if only predictions desired.
if po
  %cost = -1; %ceCost = -1; wCost = -1; numCorrect = -1;
  %grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
% cost and initial gradient can just use softmax_regression_vec, right?
% wait, theta values are the last weight layer?? so unclear... but that's what civilstat does.

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% Some error checking
assert(isempty(Z{1}), 'The first member of Z was supposed to be thrown away to normalize numbering');

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



function a = f(Z, ei)
    % hidden unit nonlinearity
    if (isequal(ei.activation_fun, 'logistic'))
        % from ex1/sigmoid.m
        a = 1 ./ (1+exp(-Z));
    else
        message = sprintf('Unknown activation function: %s\n', ei.activation_fun);
        assert(false, message);
    end
end