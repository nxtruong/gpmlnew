function [K, covdata] = covSEard(hyp, x, z, i, covdata)
% Squared Exponential covariance function with Automatic Relevance Detemination
% (ARD) distance measure. The covariance function is parameterized as:
%
% k(x^p,x^q) = sf^2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
%
% where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
% D is the dimension of the input space and sf2 is the signal variance. The
% hyperparameters are:
%
% hyp = [ log(ell_1)
%         log(ell_2)
%          .
%         log(ell_D)
%         log(sf) ]
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
% Modified and copyright (c) by Truong X. Nghiem, 2016-02-21.
%
% See also COVFUNCTIONS.M.

if nargin<2, K = '(D+1)'; return; end              % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode

[n,D] = size(x);
ell = exp(hyp(1:D));                               % characteristic length scale
sf2 = exp(2*hyp(D+1));                                         % signal variance

% precompute squared distances
if dg                                                               % vector kxx
    K = zeros(n,1);
    covdata = [];               % simple case -> don't need saved data
    has_covdata = false;        % if we have a valid covdata input
    covdata_out = false;        % if we need to save covdata output
else
    if xeqz                                                 % symmetric matrix Kxx
        % if we have a valid covdata input
        has_covdata = nargin > 4 && isequal(size(covdata), [n, n]);
        if ~has_covdata
            K = sq_dist(diag(1./ell)*x');
        end
    else                                                   % cross covariances Kxz
        nz = size(z, 1);
        % if we have a valid covdata input
        has_covdata = nargin > 4 && isequal(size(covdata), [n, nz]);
        if ~has_covdata
            K = sq_dist(diag(1./ell)*x',diag(1./ell)*z');
        end
    end
    covdata_out = nargout > 1;
end

% Covariance function data consists of this K, which is covariance matrix
if has_covdata
    K = covdata;
else
    K = sf2*exp(-K/2);                                                  % covariance
    if covdata_out
        covdata = K;
    end
end

if nargin>3 && ~isempty(i)                               % derivatives
    if i<=D                                              % length scale parameters
        if dg
            K = zeros(size(K));
        else
            if xeqz
                K = K.*sq_dist(x(:,i)'/ell(i));
            else
                K = K.*sq_dist(x(:,i)'/ell(i),z(:,i)'/ell(i));
            end
        end
    elseif i==D+1                                            % magnitude parameter
        K = 2*K;
    else
        error('Unknown hyperparameter %d', i)
    end
end