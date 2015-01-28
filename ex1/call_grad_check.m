function call_grad_check(f, theta0, data)
    % "Debugging - Gradient Checking"
    % TEMPTED to copy/paste this stuff, but my programmer's pride just won't let me...
    % f = h(data; theta) for linear/logistic regression, which are parametric models
    NUM_CHECKS = min(100, numel(theta0));%10; % check ALL partial derivatives
    
    fprintf('Checking numerical gradient...SLOW?');
    ae = grad_check(f, theta0, NUM_CHECKS, data.X, data.y);
    assert(ae < 1e-3, 'Numerical gradient fails it?');
    fprintf('\n');
end