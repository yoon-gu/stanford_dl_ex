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
    %reconstructionError = W'*z2 - x;
    %softPenalties = l1norms(z2);
    
    l1norms = sqrt(z2.^2 + params.epsilon);
    
    % not dividing by # examples because tutorial doesn't do that
    cost = params.lambda*sum(sum(l1norms));%softPenalties);% + 0.5*sum(sum(reconstructionError.^2));  
    
    %cost = params.lambda * sum(sum(sqrt(z2.^2 + params.epsilon)));
    costUnvectorized = 0;
    for i=1:size(x,2)
        for f=1:params.numFeatures
        
            % ||Wx(f,i)|| = sqrt((Wf*xi)^2 + eps)
            Wx = 0;
            for d=1:params.n
                Wx += W(f,d)*x(d,i);
            end
            s = sqrt(Wx^2 + params.epsilon);
            costUnvectorized += params.lambda * s; %sqrt(Wx^2 + params.epsilon);
            
            % need sqrt() denom
            for d=1:params.n
                Wgrad(f,d) += params.lambda * Wx * x(d,i) / s;
            end
        end
    end
    assert(abs(cost/costUnvectorized - 1) < 1e-3, 'cost vectorization')
    %size(W)
    %size(x)
    %
    %fflush(stdout);
    %disp(params.lambda * sum(l1norms(z2)) / cost);
    
    
    % unvectorized, finally correct    
    %for i = 1:size(x, 2)
    %    Wgrad = Wgrad + params.lambda*(W*x(:,i)*x(:,i)' / softPenalties(i));
    %end % vectorizing matlab code is SO annoyingly stupid...
    %Wgrad = params.lambda * z2*bsxfun(@rdivide, x, softPenalties)' + ... % this matrix derivative checks numerically
        %W*(reconstructionError)*x' + (z2)*(reconstructionError)'; % really, just copy their one-liner / 2??

%     grad = Wgrad;    
% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);

end



%function L1 = l1norms(v)
%    % returns m x 1 vector of L1-norms for each data example
%    % "In this exercise, we find epsilon = 0.01 to work well."    
%    
%    % ohhhh i have the wrong formula for L1 norm. that means the gradient is wrong too.
%        % so i had the right gradient for the WRONG FORMULA. that'll do 'er.
%    global params;
%    %L1 = sqrt(sum(v.^2, 1) + params.epsilon);
%    L1 = sum(sqrt(v.^2 + params.epsilon), 1);
%end