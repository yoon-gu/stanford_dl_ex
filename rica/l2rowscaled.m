function [y] = l2rowscaled(x, alpha)

global params;
if isfield(params, 'normeps')
    normeps = params.normeps;
else
    normeps = 1e-5;
end

epssumsq = sum(x.^2,2) + normeps;   

l2rows=sqrt(epssumsq)*alpha;
y=bsxfunwrap(@rdivide,x,l2rows);
