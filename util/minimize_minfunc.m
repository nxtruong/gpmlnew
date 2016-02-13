function [X, f, exitflag, output] = minimize_minfunc(X, f, options, varargin)
% Minimize a differentiable multivariate function using minFunc.
% (http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html)
% To be used with GPML toolbox.
%
% Usage: [X, f, exitflag, output] = ...
%               minimize_minfunc(X, f, options/length, P1, P2, ... )
% 
% X       initial guess; may be of any type, including struct and cell array
% f       the name or pointer to the function to be minimized. The function
%         f must return two arguments, the value of the function, and it's
%         partial derivatives wrt the elements of X. The partial derivative  
%         must have the same type as X.
% options options to be given to minFunc (see minFunc for details); or
% length  length of the run; if it is positive, it gives the maximum number of
%         line searches, if negative its absolute gives the maximum allowed
%         number of function evaluations. Alternatively, the length of the
%         run (and other options) can be specified by the option structure.
% P1, P2  ... parameters are passed to the function f.
%
% X       the returned solution
% f       function value at the returned solution
% exitflag returns an exit condition
% output returns a structure with other information
%        Supported Output Options
%         iterations - number of iterations taken
%         funcCount - number of function evaluations
%         algorithm - algorithm used
%         firstorderopt - first-order optimality
%         message - exit message
%         trace.funccount - function evaluations after each iteration
%         trace.fval - function value after each iteration
%
% The function returns when either its length is up, or if no further progress
% can be made (ie, we are at a (local) minimum, or so close that due to
% numerical problems, we cannot get any closer). NOTE: If the function
% terminates within a few iterations, it could be an indication that the
% function values and derivatives are not consistent (ie, there may be a bug in
% the implementation of your "f" function).
%
% Copyright (C) 2016 by Truong X. Nghiem, 2016-01-22

% Check the option
if isempty(options)
    % Use default options
    options = struct();
elseif isnumeric(options) && isscalar(options)
    if options < 0
        options = struct('MaxFunEvals', -options);
    elseif options > 0
        options = struct('MaxIter', options);
    else
        error('Run length must be non-zero.');
    end
end
assert(isstruct(options), 'Invalid option / length argument.');

% Convert objective function to function_handle
if ischar(f) && ~isempty(f)
    f = str2func(f);
end
assert(isa(f, 'function_handle'), 'Invalid objective function.');

% Remember that X and the derivative returned by f may not be vectors, so
% we must use rewrap and unwrap.
X0 = X;  % Save the structure / type of X
[X, f, exitflag, output] = minFunc(@wrapped_f, unwrap(X0), options, varargin{:});

% X is a vector, we may need to re-wrap it back to the given structure
X = rewrap(X0,X);


    function [fval, dval] = wrapped_f(cur_x, varargin)
        % This function calls unwrap() and rewrap() before and after
        % calling f
        % cur_X is a vector; dval must be a vector
        tmp = rewrap(X0, cur_x);
        [fval, dstruct] = feval(f,tmp,varargin{:});
        % dstruct can be a structure -> convert it to vector
        dval = unwrap(dstruct);
    end

end
