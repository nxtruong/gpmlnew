function [K, covdata] = covSEiso(hyp, x, z, i, varargin)
% Squared Exponential covariance function with isotropic distance measure. The
% covariance function is parameterized as:
%
% k(x^p,x^q) = sf^2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2) 
%
% where the P matrix is ell^2 times the unit matrix and sf^2 is the signal
% variance. The hyperparameters are:
%
% hyp = [ log(ell)
%         log(sf)  ]
%
% For more help on design of covariance functions, try "help covFunctions".
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
% Modified and copyright (c) by Truong X. Nghiem, 2016-01-28.
%
% See also COVFUNCTIONS.M.

if nargin<2, K = '2'; return; end                  % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode

ell = exp(hyp(1));                                 % characteristic length scale
sf2 = exp(2*hyp(2));                                           % signal variance

% precompute squared distances
% Covariance function data consists of this K
if dg                                                               % vector kxx
    K = zeros(size(x,1),1);
    covdata = [];                  % simple case -> don't need saved data
else
    nx = size(x, 1);
    if xeqz                                                 % symmetric matrix Kxx
        if ~isempty(varargin) && isequal(size(varargin{1}), [nx, nx])
            % covdata is available and valid
            K = varargin{1};
        else
            K = sq_dist(x'/ell);
        end
    else                                                   % cross covariances Kxz
        nz = size(z, 1);
        if ~isempty(varargin) && isequal(size(varargin{1}), [nx, nz])
            % covdata is available and valid
            K = varargin{1};
        else
            K = sq_dist(x'/ell,z'/ell);
        end
    end
    if nargout > 1
        covdata = K;
    end
end

if nargin<4 || isempty(i)                                          % covariances
    K = sf2*exp(-K/2);
else                                                               % derivatives
    if i==1
        K = sf2*exp(-K/2).*K;
    elseif i==2
        K = 2*sf2*exp(-K/2);
    else
        error('Unknown hyperparameter')
    end
end