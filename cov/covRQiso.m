function [K, covdata] = covRQiso(hyp, x, z, i, covdata)
% Rational Quadratic covariance function with isotropic distance measure. The
% covariance function is parameterized as:
%
% k(x^p,x^q) = sf^2 * [1 + (x^p - x^q)'*inv(P)*(x^p - x^q)/(2*alpha)]^(-alpha)
%
% where the P matrix is ell^2 times the unit matrix, sf2 is the signal
% variance and alpha is the shape parameter for the RQ covariance. The
% hyperparameters are:
%
% hyp = [ log(ell)
%         log(sf)
%         log(alpha) ]
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
% Modified and copyright (c) by Truong X. Nghiem, 2016-02-21.
%
% See also COVFUNCTIONS.M.

if nargin<2, K = '3'; return; end                  % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode

ell = exp(hyp(1));
sf2 = exp(2*hyp(2));
alpha = exp(hyp(3));

nx = size(x,1);

% precompute squared distances
% covdata is this matrix D2
% Note that one use of D2 does not multiply it with 0.5, so we should not
% precompute D2*0.5 (as in covRQard).
if dg                                                               % vector kxx
    D2 = zeros(nx,1);
    covdata_out = false;
    covdata = [];           % Simple case -> no need to save covdata
    has_covdata = false;
else
    if xeqz                                                 % symmetric matrix Kxx
        has_covdata = nargin > 4 && isequal(size(covdata), [nx, nx]);
        if ~has_covdata
            D2 = sq_dist(x'/ell);
        end
    else                                                   % cross covariances Kxz
        nz = size(z, 1);
        has_covdata = nargin > 4 && isequal(size(covdata), [nx, nz]);
        if ~has_covdata
            D2 = sq_dist(x'/ell,z'/ell);
        end
    end
    covdata_out = nargout > 1;
end

if has_covdata
    D2 = covdata;
elseif covdata_out
    covdata = D2;
end

if nargin<4  || isempty(i)                                         % covariances
    K = sf2*(1+0.5*D2/alpha).^(-alpha);
else                                                               % derivatives
    if i==1                                               % length scale parameter
        K = sf2*(1+0.5*D2/alpha).^(-alpha-1).*D2;
    elseif i==2                                              % magnitude parameter
        K = 2*sf2*(1+0.5*D2/alpha).^(-alpha);
    elseif i==3
        K = (1+0.5*D2/alpha);
        K = sf2*K.^(-alpha).*(0.5*D2./K - alpha*log(K));
    else
        error('Unknown hyperparameter')
    end
end