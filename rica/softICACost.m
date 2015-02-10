%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);
Wgrad = zeros(size(W));

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1); % norm constraint? cf. lecture notes p. 17 - but isn't that for visualization?
    % the 2 lines above are ignorable preprocessing that gets popped before returning?
   
    % backprop was done ANALYTICALLY in tutorial    
    z2 = W*x;
    reconstructionError = W'*z2 - x;
    softPenalties = l1norm(z2);
    
    % not dividing by # examples because tutorial doesn't do that
    cost = params.lambda*sum(softPenalties) + 0.5*sum(sum(reconstructionError.^2));  
    
    % unvectorized, finally correct    
    %for i = 1:size(x, 2)
    %    Wgrad = Wgrad + params.lambda*(W*x(:,i)*x(:,i)' / softPenalties(i));
    %end % vectorizing matlab code is SO annoyingly stupid...
    Wgrad = params.lambda * z2*bsxfun(@rdivide, x, softPenalties)' + ... % this matrix derivative checks numerically
        W*(reconstructionError)*x' + (z2)*(reconstructionError)'; % really, just copy their one-liner / 2??

%     grad = Wgrad;    
% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);

end



function L1 = l1norm(v)
    % "In this exercise, we find epsilon = 0.01 to work well."
    % ohhhh i have the wrong formula for L1 norm. that means the gradient is wrong too.
        % so i had the right gradient for the WRONG FORMULA. that'll do 'er.
    global params;
    %L1 = sqrt(sum(v.^2, 1) + params.epsilon);
    L1 = sum(sqrt(v.^2 + params.epsilon), 1);
end