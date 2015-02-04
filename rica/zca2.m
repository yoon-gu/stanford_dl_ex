function [Z, V] = zca2(x, epsilon)
    if nargin < 2
        epsilon = 1e-4; % ugh, ICA section p. 2 mentions this BRIEFLY. but only orthonormal ICA requires epsilon = 0, not RICA, right?
    end
% You should be able to use the code from your PCA/ZCA exercise
% Retain all of the components from the ZCA transform (i.e. do not do
% dimensionality reduction)

% x is the input patch data of size
% z is the ZCA transformed data. The dimenison of z = x.

    % following pca_gen.m
    avg = mean(x, 1); % can't ignore for unnatural images like digits?
    %avg = mean(x, 2); % both removeDC.m and ICAExercise.m have this, but I think it's wrong. at least, it doesn't match 4b.pdf or 4c.pdf
    
    x = x - repmat(avg, size(x, 1), 1); % their storage-hungry version
    %x = bsxfun(@minus, x, avg);
    
    Sigma = x * x' / size(x, 2);
    [U, S, V] = svd(Sigma);
    
    %xRot = U' * x;
    %xPCAWhite = bsxfun(@rdivide, xRot, sqrt(diag(S) + epsilon));
    %Z = U * xPCAWhite; % xZCAWhite
    
    Z = U * diag(1./sqrt(diag(S) + epsilon)) * U' * x;
    %Z = U * bsxfun(@rdivide, U' * x, sqrt(diag(S) + epsilon))
