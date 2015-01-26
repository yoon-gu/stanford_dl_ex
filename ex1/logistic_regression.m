function [f,g] = logistic_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %

  m=size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));


  %
  %        Compute the objective function by looping over the dataset and summing
  %        up the objective values for each example.  Store the result in 'f'.
  %
  %        Compute the gradient of the objective by looping over the dataset and summing
  %        up the gradients (df/dtheta) for each example. Store the result in 'g'.
  %
    for i=1:m
        xi = X(:,i);
        yi = y(i);
        hi = sigmoid(theta' * xi);
        
        f = f - (yi*log(hi) + (1-yi)*log(1-hi));
        g = g + xi*(hi-yi); % this is already "slow enough" - no need to "unvectorize" further...
    end
