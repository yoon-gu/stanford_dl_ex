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
    reconstructionError = W'*z2 - x;
    l1norms = sqrt(z2.^2 + params.epsilon);
    
    % not dividing by # examples because tutorial doesn't do that
    cost = params.lambda*sum(sum(l1norms)) + 0.5*sum(sum(reconstructionError.^2));  
    Wgrad = params.lambda * (z2 ./ l1norms) * x' + ...
        W*(reconstructionError)*x' + (z2)*(reconstructionError)'; % really, just copy their one-liner / 2??
    
    DEBUG = false;
    if DEBUG
        costUnvectorized = 0;
        WgradUnvectorized = zeros(size(W));
        for i=1:size(x,2)
            for f=1:params.numFeatures
            
                % ||Wx(f,i)|| = sqrt((Wf*xi)^2 + eps)
                Wx = 0;
                for d=1:params.n
                    Wx = Wx + W(f,d)*x(d,i);
                end
                s = sqrt(Wx^2 + params.epsilon);
                costUnvectorized = costUnvectorized + params.lambda * s; 
                
                % need sqrt() denom
                for d=1:params.n
                    WgradUnvectorized(f,d) = WgradUnvectorized(f,d) + ...
                        params.lambda * Wx * x(d,i) / s;
                end
            end
        end
        assert(abs(cost/costUnvectorized - 1) < 1e-3, 'cost vectorization')
        assert(norm(abs(Wgrad - WgradUnvectorized)) < 1e-3, 'gradient vectorization')
    end

%     grad = Wgrad;    
% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);

end

