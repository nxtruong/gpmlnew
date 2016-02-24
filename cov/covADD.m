function [K, covdata] = covADD(cov, hyp, x, z, i, covdata)
% Additive covariance function using 1d base covariance functions
% cov_d(x_d,z_d;hyp_d) with individual hyperparameters hyp_d, d=1..D.
%
% k  (x,z) = \sum_{r \in R} sf^2_r k_r(x,z), where 1<=r<=D and
% k_r(x,z) = \sum_{|I|=r} \prod_{i \in I} cov_i(x_i,z_i;hyp_i)
%
% hyp = [ hyp_1
%         hyp_2
%          ...
%         hyp_D
%         log(sf_R(1))
%          ...
%         log(sf_R(end)) ]
%
% where hyp_d are the parameters of the 1d covariance function which are shared
% over the different values of r=R(1),..,R(end) where 1<=r<=D.
%
% Usage: covADD({[1,3,4],cov}, ...) or
%        covADD({[1,3,4],cov_1,..,cov_D}, ...).
%
% Please see the paper "Additive Gaussian Processes" by Duvenaud, Nickisch and
% Rasmussen, NIPS, 2011 for details.
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2016-01-27.
%                  multiple covariance support contributed by Truong X. Nghiem
%
% See also COVFUNCTIONS.M.

R = fix(cov{1});                            % only positive integers are allowed
if min(R)<1, error('only positive R up to D allowed'), end
nr = numel(R);                      % number of different degrees of interaction
nc = numel(cov)-1;                              % number of provided covariances
for j=1:nc, if ~iscell(cov{j+1}), cov{j+1} = cov(j+1); end, end
if nc==1, nh = eval(feval(cov{2}{:}));  % no of hypers per individual covariance
else nh = zeros(nc,1); for j=1:nc, nh(j) = eval(feval(cov{j+1}{:})); end, end

if nargin<3                                  % report number of hyper parameters
    if nc==1, K = ['D*', int2str(nh), '+', int2str(nr)];
    else
        K = ['(',int2str(nh(1))]; for ii=2:nc, K = [K,'+',int2str(nh(ii))]; end
        K = [K, ')+', int2str(nr)];
    end
    return
end
if nargin<4, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode

D = size(x,2);                                                % dimensionality
if nc==1
    nh = ones(D,1)*nh;
    cov = [cov(1),repmat(cov(2),1,D)];
elseif nc ~= D
    error('each dimension must have exactly one covariance function.');
end
nch = sum(nh);                                      % total number of cov hypers
% sf2 is involved in computation in the 3rd dimension, so we make it 3D to
% use it with bsxfun().
sf2 = reshape(exp( 2*hyp(nch+(1:nr)) ), 1, 1, []);         % signal variances of individual degrees

% Covariance function data consists of:
% Kd (below)
% EE (below)
% Ks: cell array of K (as computed in the case i <= nch below) for each
%       dimension j.
% covdatas: cell array of covdata's for each dimension
has_covdata = nargin > 5 && isstruct(covdata);
covdata_out = nargout > 1;

if nargin<5 || isempty(i)                                          % covariances
    EE = getEE();
    % sf2 weighted sum
    %K = 0; for ii=1:nr, K = K + sf2(ii)*EE(:,:,R(ii)+1); end
    K = sum(bsxfun(@times, sf2, EE(:,:,R+1)), 3);
else                                                               % derivatives
    if i <= nch                        % individual covariance function parameters
        j = find(cumsum(nh)>=i,1,'first');  % j, the dimension of the hyperparameter
        if dg, zj = 'diag'; else if xeqz, zj = []; else zj = z(:,j); end, end
        cj = cov{j+1}; nchj = sum(nh(1:j-1)); hypj = hyp(nchj+(1:nh(j)));
        if has_covdata && isfield(covdata, 'covdatas')
            [dKj, covdata.covdatas{j}] = feval(cj{:},hypj,x(:,j),zj,i-nchj,covdata.covdatas{j});  % other dK=0
        else
            dKj = feval(cj{:},hypj,x(:,j),zj,i-nchj);  % other dK=0
        end
        % the final derivative is a sum of multilinear terms, so if only one term
        % depends on the hyperparameter under consideration, we can factorise it
        % out and compute the sum with one degree less
        if has_covdata && isfield(covdata, 'Ks') && ~isempty(covdata.Ks{j})
            K = dKj.*covdata.Ks{j};
        else
            Kd = getKd();
            E = elsympol(Kd(:,:,[1:j-1,j+1:D]),max(R)-1);  %  R-1th elementary sym polyn
            %K = 0; for ii=1:nr, K = K + sf2(ii)*E(:,:,R(ii)); end     % sf2 weighted sum
            K = sum(bsxfun(@times, sf2, E(:,:,R)), 3);
            if covdata_out
                if ~isfield(covdata, 'Ks'), covdata.Ks = cell(1,D); end
                covdata.Ks{j} = K;  % save K of the j-th dimension
            end
            K = dKj.*K;
        end
    elseif i <= nch+nr
        EE = getEE();
        j = i-nch;
        K = 2*sf2(j)*EE(:,:,R(j)+1);                  % rest of the sf2 weighted sum
    else
        error('Unknown hyperparameter')
    end
end

    function Kd = getKd
        % Returns Kd, either from covdata or computing it (and save it to
        % covdata).
        if has_covdata
            % if covdata exists, it must at least contain Kd
            Kd = covdata.Kd;
        else
            if covdata_out
                covdata = struct();
                % evaluate dimensionwise covariances K and save covdata's
                [Kd, covdata.covdatas] = Kdim(cov(2:end),nh,hyp(1:nch),x,z);
                covdata.Kd = Kd;
            else
                Kd = Kdim(cov(2:end),nh,hyp(1:nch),x,z);  % evaluate dimensionwise covariances K
            end
        end
    end

    function EE = getEE
        % Returns EE, either from covdata or computing it (and save it to
        % covdata).
        if has_covdata && isfield(covdata, 'EE')
            EE = covdata.EE;
        else
            EE = elsympol(getKd(),max(R));             % Rth elementary symmetric polynomials
            if covdata_out
                covdata.EE = EE;
            end
        end
    end
end

% evaluate dimensionwise covariances K
function [K, dcovdata] = Kdim(cov,nh,hyp,x,z)
% dcovdata is a cell array of covdata's of the covariance functions of each
% dimension.
[n,D] = size(x);                                              % dimensionality
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;      % determine mode

if dg                                                        % allocate memory
    K = zeros(n,1,D);
else
    if xeqz, K = zeros(n,n,D); else K = zeros(n,size(z,1),D); end
end
dcovdata = cell(1, D);

for d=1:D
    hyp_d = hyp(sum(nh(1:d-1))+(1:nh(d)));        % hyperparamter of dimension d
    if dg
        [K(:,:,d), dcovdata{d}] = feval(cov{d}{:},hyp_d,x(:,d),'diag');
    else
        if xeqz
            [K(:,:,d), dcovdata{d}] = feval(cov{d}{:},hyp_d,x(:,d));
        else
            [K(:,:,d), dcovdata{d}] = feval(cov{d}{:},hyp_d,x(:,d),z(:,d));
        end
    end
end
end