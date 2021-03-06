function [K, covdata] = covDiscrete(s, hyp, x, z, i, covdata)

% Covariance function for discrete inputs. Given a function defined on the
% integers 1,2,3,..,s, the covariance function is parameterized as:
%
% k(x,z) = K_{xz},
%
% where K is a matrix of size (s x s).
%
% This implementation assumes that the inputs x and z are given as integers
% between 1 and s, which simply index the matrix K.
%
% The hyperparameters specify the upper-triangular part of the Cholesky factor
% of K, where the (positive) diagonal elements are specified by their logarithm.
% If L = chol(K), so K = L'*L, then the hyperparameters are:
%
% hyp = [ log(L_11)
%             L_21
%         log(L_22)
%             L_31
%             L_32
%         ..
%         log(L_ss) ]
%
% The hyperparameters hyp can be generated from K using:
% L = chol(K); L(1:(s+1):end) = log(diag(L));
% hyp = L(triu(true(s)));
%
% The covariance matrix K is obtained from the hyperparameters hyp by:
% L = zeros(s); L(triu(true(s))) = hyp(:);
% L(1:(s+1):end) = exp(diag(L)); K = L'*L;
%
% This parametrization allows unconstrained optimization of K.
%
% For more help on design of covariance functions, try "help covFunctions".
%
% Copyright (c) by Roman Garnett, 2014-08-14.
% Modified by Truong X. Nghiem, 2016-04-01.
%
% See also MEANDISCRETE.M, COVFUNCTIONS.M.

if nargin==0, error('s must be specified.'), end           % check for dimension
if nargin<3, K = num2str(s*(s+1)/2); return; end   % report number of parameters
if nargin<4, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode
if xeqz, z = x; end                                % make sure we have a valid z

covdata = [];

L = zeros(s); L(triu(true(s))) = hyp(:);                 % build Cholesky factor
L(1:(s+1):end) = exp(diag(L));

if nargin<5 || isempty(i)
  A = L'*L; % A is a placeholder for K to avoid a name clash with the return arg
  if dg
    K = A(sub2ind(size(A),x,x));
  else
    K = A(x,z);
  end
else
  col = ceil((sqrt(8*i+1)-1)/2); row = i-col*(col-1)/2;    % indices by tri-root
  dL = zeros(s);                                 % derivative of Cholesky factor
  if (row == col)
    dL(row,col) = L(row,col);         % diagonal entries have exp transformation
  else
    dL(row,col) = 1;
  end
  dK = dL'*L; dK = dK+dK';                 % derivative of sxs covariance matrix
  if dg
    K = dK(sub2ind(size(dK),x,x));
  else
    K = dK(x,z);
  end
end