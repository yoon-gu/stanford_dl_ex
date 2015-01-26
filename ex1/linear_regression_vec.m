function [f,g] = linear_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example.  y(j) is the target for example j.
  %
  m=size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  %        Compute the linear regression objective function and gradient 
  %        using vectorized code.  (It will be just a few lines of code!)           % linear_regression.m wasn't THAT much longer...
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %
    % annoyingly, X is the transpose of the usual design matrix
    errors = theta'*X - y;
    f = 0.5 * sum(errors.^2);
    
    % i think this is right by inspection
    g = X * errors'; 
