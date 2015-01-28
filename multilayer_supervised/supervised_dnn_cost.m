function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)
    % gah? ceCost, wCost return values are defunct...

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
hAct = cell(numHidden+1, 1);
gradStack = params2stack(grad, ei);

%% forward prop
%%% YOUR CODE HERE %%%

%% return here if only predictions desired.
if po
  %cost = -1; %ceCost = -1; wCost = -1; numCorrect = -1;
  %grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



