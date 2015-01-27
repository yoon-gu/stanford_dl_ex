function [f,g] = softmax_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  %        Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
  
    % Since the cost function is ONLY evaluated here, it's entirely arbitrary 
    % which class for which to have theta = 0. Follow the tutorial and set theta(K) = 0.
    % I guess if you really want performance, you can comment out all the asserts...
    assert(isequal(size(theta), [n num_classes-1]))
    fulltheta = [theta zeros(n, 1)];
    
    % not wasteful - must construct all terms for the partition functions.
    % p_unnormalized(k,i) = unnormalized P(y(i) = k)
    p_unnormalized = exp(fulltheta' * X); 
    
    % normalize and take logarithms
    P = bsxfun(@rdivide, p_unnormalized, sum(p_unnormalized, 1)); % bsxfun from hint 2
    assert(norm(sum(P, 1) - 1) < 1e-6); % there's gonna be floating point differences
    logP = log(P);
    
    % negative log likelihood of the data - for canned minimizer
    assert(all(1 <= y) && all(y <= num_classes)); % since MATLAB indexing is 1-based
    assert(m == size(P, 2));
    f = -sum(logP( sub2ind(size(P), y, 1:m) )); % sub2ind from hint 2
    
    
    
    % various testing during development
    DEBUG = true;
    if DEBUG
        
        % looks like when there's no test code in place, the easiest thing to do 
        % is just to check right here with unvectorized versions
        for i=1:m
            for k=1:num_classes
                pu(k,i) = exp(fulltheta(:, k)' * X(:, i));
            end            
            %assert(isequal(p_unnormalized(:, i), exp(fulltheta' * X(:, i))));
        end
        assert(norm(p_unnormalized - pu) < 1e-6);
        
        f_debug = 0;
        for i=1:m
            f_debug = f_debug - logP(y(i), i);
        end
        assert(abs(f - f_debug) < 1e-6);
        
        g_debug = zeros(size(theta));
       
    else
        % due to memory crashes, did debugging with smaller testing data set only
        assert(m <= 10000, 'Did you forget to restore the real training data?')
    end
    
  
  g=g(:); % make gradient a vector for minFunc

