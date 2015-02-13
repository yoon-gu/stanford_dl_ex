function [Z, U, S, V] = zca2(x, epsilon)
    if nargin < 2       % see also old tutorial's section "Data Preprocessing"
        epsilon = 1e-4; % ugh, ICA section p. 2 mentions this BRIEFLY. but only orthonormal ICA requires epsilon = 0, not RICA, right?
    end
% You should be able to use the code from your PCA/ZCA exercise
% Retain all of the components from the ZCA transform (i.e. do not do
% dimensionality reduction)

% x is the input patch data of size
% z is the ZCA transformed data. The dimenison of z = x.

    if ~isOctave()      % Octave: single() results in illegal step direction later!?!?
%         x = single(x); % 32-bit MATLAB: otherwise run out of memory... FAST!? (~90 sec for RICA training??)
    end

    % following pca_gen.m
    avg = mean(x, 1); % can't ignore for unnatural images like digits?
    %avg = mean(x, 2); % both removeDC.m and ICAExercise.m have this, but I think it's wrong. at least, it doesn't match tutorial's PCA section...
    
    x = x - repmat(avg, size(x, 1), 1); % their storage-hungry version
    %x = bsxfun(@minus, x, avg);
%     stride = 1000;
%     assert(mod(size(x,2), stride) == 0)
%     for i=1:stride:size(x,2)
%         block = i : i+stride-1;
%         %x(:, block) = bsxfun(@minus, x(:, block), avg(block));
%         x(:, block) = x(:, block) - repmat(avg(block), size(x, 1), 1);
%     end
    
    Sigma = x * x' / size(x, 2);
    [U, S, V] = svd(Sigma);
    
    %xRot = U' * x;
    %xPCAWhite = bsxfun(@rdivide, xRot, sqrt(diag(S) + epsilon));
    %Z = U * xPCAWhite; % xZCAWhite
    
    Z = U * diag(1./sqrt(diag(S) + epsilon)) * U' * x; % ironically, this seems more memory-frugal
    %Z = U * bsxfun(@rdivide, U' * x, sqrt(diag(S) + epsilon))
