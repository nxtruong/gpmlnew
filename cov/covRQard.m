function [K, covdata] = covRQard(hyp, x, z, i, covdata)
% Rational Quadratic covariance function with Automatic Relevance Determination
% (ARD) distance measure. The covariance function is parameterized as:
%
% k(x^p,x^q) = sf^2 * [1 + (x^p - x^q)'*inv(P)*(x^p - x^q)/(2*alpha)]^(-alpha)
%
% where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
% D is the dimension of the input space, sf2 is the signal variance and alpha
% is the shape parameter for the RQ covariance. The hyperparameters are:
%
% hyp = [ log(ell_1)
%         log(ell_2)
%          ..
%         log(ell_D)
%         log(sf)
%         log(alpha) ]
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-08-04.
% Modified and copyright (c) by Truong X. Nghiem, 2016-02-21.
%
% See also COVFUNCTIONS.M.

if nargin<2, K = '(D+2)'; return; end              % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode

[n,D] = size(x);
ell = exp(hyp(1:D));
sf2 = exp(2*hyp(D+1));
alpha = exp(hyp(D+2));

% precompute squared distances
% Because D2 is always multiplied by 0.5 later on, we will compute D2*0.5
% covdata is this matrix
if dg                                                               % vector kxx
    D2 = zeros(n,1);
    covdata_out = false;
    covdata = [];           % Simple case -> no need to save covdata
    has_covdata = false;
else
    if xeqz                                                 % symmetric matrix Kxx
        has_covdata = nargin > 4 && isequal(size(covdata), [n, n]);
        if ~has_covdata
            D2 = sq_dist(diag(1./ell)*x') * 0.5;
        end
    else                                                   % cross covariances Kxz
        nz = size(z, 1);
        has_covdata = nargin > 4 && isequal(size(covdata), [n, nz]);
        if ~has_covdata
            D2 = sq_dist(diag(1./ell)*x',diag(1./ell)*z') * 0.5;
        end
    end
    covdata_out = nargout > 1;
end

if has_covdata
    D2 = covdata;
elseif covdata_out
    covdata = D2;
end

if nargin<4 || isempty(i)                                       % covariances
    K = sf2*(1+D2/alpha).^(-alpha);
else                                                               % derivatives
    if i<=D                                               % length scale parameter
        if dg
            K = zeros(size(D2));
        else
            if xeqz
                K = sf2*(1+D2/alpha).^(-alpha-1).*sq_dist(x(:,i)'/ell(i));
            else
                K = sf2*(1+D2/alpha).^(-alpha-1).*sq_dist(x(:,i)'/ell(i),z(:,i)'/ell(i));
            end
        end
    elseif i==D+1                                            % magnitude parameter
        K = 2*sf2*(1+D2/alpha).^(-alpha);
    elseif i==D+2
        K = (1+D2/alpha);
        K = sf2*K.^(-alpha).*(D2./K - alpha*log(K));
    else
        error('Unknown hyperparameter')
    end
end