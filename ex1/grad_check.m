function average_error = grad_check(fun, theta0, num_checks, varargin)
  % apparently this (error-riddled) function numerically checks a random sample of partial derivatives.
  % it doesn't even check the full gradient unless you take num_checks = numel(theta0)

  delta=1e-3; 
  sum_error=0;

  fprintf(' Iter       i             err');
  fprintf('           g_est               g               f\n')

  T = theta0;
  [f,g] = fun(T, varargin{:});
  
  shuffle = randperm(numel(T));
  
  for i=1:num_checks
    %T = theta0;
    j = shuffle(i);%randsample(numel(T),1);
    T0=T; T0(j) = T0(j)-delta;
    T1=T; T1(j) = T1(j)+delta;

    %[f,g] = fun(T, varargin{:}); % WHY are you repeatedly calling lines like this in the inner loop??
    f0 = fun(T0, varargin{:});
    f1 = fun(T1, varargin{:});

    g_est = (f1-f0) / (2*delta);
    error = abs(g(j) - g_est);

    fprintf('% 5d  % 6d % 15g % 15f % 15f % 15f\n', ...
            i,j,error,g(j),g_est,f);

    sum_error = sum_error + error;
  end

  average_error=sum_error/num_checks; % wow, another mistake... whoever wrote this is not very careful...
end



function y = randsample(n, k)
    % returns a k-by-1 vector y of values sampled uniformly at random, without replacement, from the integers 1 to n.
    % undefined in Octave 3.2.4
    if isOctave
        y = randperm(n);
        y = y(1:k);
    else
        y = randsample(n, k);
    end
end
