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
    %p_unnormalized = exp(fulltheta' * X); 
    p_unnormalized = [exp(theta' * X); ones(1, m)]; % 10% fewer exp() calls - optimizing for neural network
    
    % normalize and take logarithms. store log(P) for f; store P for g (gradient)
    P = bsxfun(@rdivide, p_unnormalized, sum(p_unnormalized, 1)); % bsxfun from hint 2
    %assert(norm(sum(P, 1) - 1) < 1e-3); % there's gonna be floating point differences - failing later in optimization with useMex... large weights?
    logP = log(P);
    
    % negative log likelihood of the data - for canned minimizer
    assert(all(1 <= y) && all(y <= num_classes)); % since MATLAB indexing is 1-based
    assert(m == size(P, 2));
    f = -sum(logP( sub2ind(size(P), y, 1:m) )); % sub2ind from hint 2

    % gradient of negative log likelihood.
    % sub2ind on y=class won't work here because it's X(dimension, example), unlike P(class, example).
    % is there any slicker way to do this? check github? (well, i mean, it's 4 lines already...)
        % https://github.com/civilstat/stanford_dl_ex/blob/master/ex1/softmax_regression_vec.m - nice!
        % looks like things are a little more vectorized if you construct a coefficient matrix with sub2ind 
        % before multiplying by P. also, you can cut down the number of exp() calls by 10% by hand-constructing
        % the last row of p_unnormalized to be just ones, instead of "fulltheta".
        % civilstat has a much better understanding of how to interpret these statistical quantities:
          % """
          % Compute "residuals": for each class and j'th sample, we get 
          %   p_hat if j'th sample is not that class (y(j) = 0), or
          %   (p_hat-1) if j'th sample is that class (y(j) = 1),
          %   so should be small if p_hat is good
          % """
        % as I recall my maxent, it's "expected value - observed value"
    g = X * P(1:num_classes-1, :)';
    for k=1:num_classes-1
        g(:, k) = g(:, k) - sum(X(:, y == k), 2);
    end
    
    % various testing during development
    DEBUG = false;
    if DEBUG
        
        % looks like when there's no test code in place, the easiest thing to do 
        % is just to check right here with unvectorized versions
        for i=1:m
            for k=1:num_classes
                pu(k,i) = exp(fulltheta(:, k)' * X(:, i));
            end            
            %assert(isequal(p_unnormalized(:, i), exp(fulltheta' * X(:, i))));
        end
        
        pass = mean(abs(p_unnormalized(:) - pu(:))) < 1e-3 * max(pu(:)); % large signal can have large unnormalized p
        if ~pass
            mean(abs(p_unnormalized(:) - pu(:)))
            [mymax, mymaxindex] = max(abs(p_unnormalized(:) - pu(:)));
            p_unnormalized(mymaxindex)
            pu(mymaxindex)
            p_unnormalized - pu
        end
        assert(pass, 'probability vectorization failed!?');
        
        f_debug = 0;
        for i=1:m
            f_debug = f_debug - logP(y(i), i);
        end
        assert(abs(f - f_debug) < 1e-6, 'cost function vectorization failed!?');
        
        g_debug = zeros(size(theta));
        for i=1:m
            %fprintf('%d/%d\n', i, m);
            %fflush(stdout);
            for k=1:num_classes-1                
                coeff = -P(k, i);                
                if (y(i) == k)
                    coeff = coeff + 1;
                end
                g_debug(:, k) = g_debug(:, k) - X(:, i) * coeff;
            end
        end
        assert(norm(g_debug - g) < 1e-6, 'gradient vectorization failed?');
       
    else
        % due to memory crashes, did debugging with smaller testing data set only
        %assert(m > 10000, 'Did you forget to restore the real training data?')
    end
    
  
  g=g(:); % make gradient a vector for minFunc

