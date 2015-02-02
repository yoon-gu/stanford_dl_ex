%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);
Wgrad = zeros(size(W));

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);
    % the 2 lines above are ignorable preprocessing that gets popped before returning?

    % backprop was done ANALYTICALLY in tutorial    
    %size(W) % 50 features x 81 pixels
    %size(x) % 81 pixels x 10000 examples (patches)
    %params    
    %m = size(x, 2);
    reconstructionError = W'*W*x - x;
    cost = 0.5*sum(sum(reconstructionError.^2));  % not dividing by # examples because tutorial doesn't
    Wgrad = W*(reconstructionError)*x' + (W*x)*(reconstructionError)'; % really, just copy their one-liner / 2??

% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);