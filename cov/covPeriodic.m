function [K, covdata] = covPeriodic(hyp, x, z, i, covdatain)
% Stationary covariance function for a smooth periodic function, with period p
% in 1d (see covPERiso and covPERard for multivariate data):
%
% k(x,z) = sf^2 * exp( -2*sin^2( pi*||x-z||/p )/ell^2 )
%
% where the hyperparameters are:
%
% hyp = [ log(ell)
%         log(p)
%         log(sf) ]
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2011-01-05.
% Modified and copyright (c) by Truong X. Nghiem, 2016-01-28.
%
% See also COVFUNCTIONS.M.

if nargin<2, K = '3'; return; end                  % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode

[n,D] = size(x);
if D>1, error('Covariance is defined for 1d data only.'), end
ell = exp(hyp(1));
p   = exp(hyp(2));
sf2 = exp(2*hyp(3));

% covdata is a structure of K (precomputed below) and R = sin(K)/ell

% precompute distances
if dg                                                               % vector kxx
    K = zeros(size(x,1),1);
    R = K;
    covdata = struct('K', [], 'R', []);     % simple case -> don't need saved data
else
    nx = size(x, 1);
    has_covdata = (nargin > 4) && isstruct(covdatain) ...
        && all(isfield(covdatain, {'K', 'R'}));
    if xeqz                                                 % symmetric matrix Kxx
        if has_covdata && isequal(size(covdatain.K), [nx, nx]) ...
                && isequal(size(covdatain.R), [nx, nx])
            % covdata is available and valid
            K = covdatain.K;
            R = covdatain.R;
        else
            K = (pi/p)*sqrt(sq_dist(x'));
            R = sin(K)/ell;
        end
    else                                                   % cross covariances Kxz
        nz = size(z, 1);
        if has_covdata && isequal(size(covdatain.K), [nx, nz]) ...
                && isequal(size(covdatain.R), [nx, nz])
            % covdata is available and valid
            K = covdatain.K;
            R = covdatain.R;
        else
            K = (pi/p)*sqrt(sq_dist(x',z'));
            R = sin(K)/ell;
        end
    end
    if nargout > 1
        covdata = struct('K', K, 'R', R);
    end
end

if nargin<4 || isempty(i)                                          % covariances
    K = sf2*exp(-2*(R.^2));
else                                                               % derivatives
    if i==1
        R = R.^2; K = 4*sf2*exp(-2*R).*R;
    elseif i==2
        K = 4*sf2/ell*exp(-2*(R.^2)).*R.*cos(K).*K;
    elseif i==3
        K = 2*sf2*exp(-2*(R.^2));
    else
        error('Unknown hyperparameter')
    end
end