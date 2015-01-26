function [f,g] = linear_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example.  y(j) is the target for example j.
  %
  
  m=size(X,2);
  n=size(X,1);

  f=0;
  g=zeros(size(theta));

  %
  %        Compute the linear regression objective by looping over the examples in X.
  %        Store the objective function value in 'f'.
  %
  %        Compute the gradient of the objective with respect to theta by looping over
  %        the examples in X and adding up the gradient for each example.  Store the
  %        computed gradient in 'g'.
  
    % unvectorized for runtime comparison
    for i=1:m
        xi = X(:, i);
        error = theta'*xi - y(i);
        
        f = f + error^2;
        
        % g = g + xi*error; % uh, i had FIGHT my instincts to write the loop below
        for j = 1:n
            g(j) = g(j) + xi(j)*error;
        end
        
    end
    
    f = 0.5*f;
  

  

