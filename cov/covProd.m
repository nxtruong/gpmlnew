function [K, covdata] = covProd(cov, hyp, x, z, i, covdatain)
% covProd - compose a covariance function as the product of other covariance
% functions. This function doesn't actually compute very much on its own, it
% merely does some bookkeeping, and calls other covariance functions to do the
% actual work.
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
% Modified and copyright (c) by Truong X. Nghiem, 2016-01-28.
%
% See also COVFUNCTIONS.M.

if isempty(cov), error('We require at least one factor.'), end
j = cell(1, numel(cov));
for ii = 1:numel(cov)                        % iterate over covariance functions
    f = cov(ii); if iscell(f{:}), f = f{:}; end   % expand cell array if necessary
    j{ii} = feval(f{:});                          % collect number hypers
end

if nargin<3                                        % report number of parameters
    K = j{1}; for ii=2:length(j), K = [K, '+', j{ii}]; end, return
end
if nargin<4, z = []; end                                   % make sure, z exists
[n,D] = size(x);

v = [];               % v vector indicates to which covariance parameters belong
for ii = 1:length(j), v(end+1:end+eval(j{ii})) = ii; end

% covdata is a structure array of length length(cov) to save both the
% covdata 'd' and covariance matrix 'K' of each covariance function.

covdata_out = nargout > 1;

if nargin<5 || isempty(i)                                          % covariances
    % Because the covariances are only computed once in each iteration, we
    % don't need to save it.
    
    K = 1;                                                                  % init
    if covdata_out
        % The saved data of covariance functions
        covdata = repmat(struct('d', [], 'K', []), length(cov), 1);
    end
    for ii = 1:length(cov)                       % iteration over factor functions
        f = cov(ii); if iscell(f{:}), f = f{:}; end % expand cell array if necessary
        if covdata_out
            [Kii, covdata(ii).d] = feval(f{:}, hyp(v==ii), x, z);
            covdata(ii).K = Kii;
        else
            Kii = feval(f{:}, hyp(v==ii), x, z);
        end
        K = K .* Kii;             % accumulate covariances
    end
else                                                               % derivatives
    if nargin > 5
        has_covdata =  isstruct(covdatain) && (numel(covdatain) == length(cov)) ...
            && all(isfield(covdatain, {'d', 'K'}));
    else
        has_covdata = false;
    end
    
    if i<=length(v)
        if covdata_out
            if has_covdata
                covdata = covdatain;
            else
                covdata = repmat(struct('d', [], 'K', []), length(cov), 1);
            end
        end
        K = 1; vi = v(i);                                % which covariance function
        j = sum(v(1:i)==vi);                    % which parameter in that covariance
        for ii = 1:length(cov)                     % iteration over factor functions
            f = cov(ii); if iscell(f{:}), f = f{:}; end     % expand cell if necessary
            if ii==vi
                % accumulate covariances
                if has_covdata
                    if covdata_out
                        % Note that we don't save Kii to covdata(ii).K because it's
                        % not the covariance matrix
                        [Kii, covdata(ii).d] = feval(f{:}, hyp(v==ii), x, z, j, covdatain(ii).d);
                    else
                        Kii = feval(f{:}, hyp(v==ii), x, z, j, covdatain(ii).d);
                    end
                else
                    if covdata_out
                        % Note that we don't save Kii to covdata(ii).K because it's
                        % not the covariance matrix
                        [Kii, covdata(ii).d] = feval(f{:}, hyp(v==ii), x, z, j);
                    else
                        Kii = feval(f{:}, hyp(v==ii), x, z, j);
                    end
                end
            else
                if has_covdata && ~isempty(covdatain(ii).K)
                    % The covariance matrix is already available, reuse it
                    Kii = covdatain(ii).K;
                else
                    % We need to call the cov function to compute Kii
                    [Kii, covdataii] = feval(f{:}, hyp(v==ii), x, z);
                    if covdata_out
                        covdata(ii).d = covdataii;
                        covdata(ii).K = Kii;
                    end
                end
            end
            K = K .* Kii;
        end
    else
        error('Unknown hyperparameter')
    end
end