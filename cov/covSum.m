function [K, covdata] = covSum(cov, hyp, x, z, i, varargin)
% covSum - compose a covariance function as the sum of other covariance
% functions. This function doesn't actually compute very much on its own, it
% merely does some bookkeeping, and calls other covariance functions to do the
% actual work.
%
% Copyright (c) by Carl Edward Rasmussen & Hannes Nickisch 2010-09-10.
% Modified and copyright (c) by Truong X. Nghiem, 2016-01-28.
%
% See also COVFUNCTIONS.M.

if ~isempty(cov)==0, error('We require at least one summand.'), end
for ii = 1:numel(cov)                        % iterate over covariance functions
    f = cov(ii); if iscell(f{:}), f = f{:}; end   % expand cell array if necessary
    j(ii) = cellstr(feval(f{:}));                          % collect number hypers
end

if nargin<3                                        % report number of parameters
    K = char(j(1)); for ii=2:length(cov), K = [K, '+', char(j(ii))]; end, return
end
if nargin<4, z = []; end                                   % make sure, z exists
[n,D] = size(x);

v = [];               % v vector indicates to which covariance parameters belong
for ii = 1:length(cov), v = [v repmat(ii, 1, eval(char(j(ii))))]; end

has_covdata = ~isempty(varargin) && iscell(varargin{1}) && ...
    (numel(varargin{1}) == length(cov)+1);
covdata_out = nargout > 1;

if nargin<5 || isempty(i)                                          % covariances
    if has_covdata
        % We have saved data -> use K immediately
        K = varargin{1}{end};
        if covdata_out
            covdata = varargin{1};
        end
        return;
    end
    
    K = 0;
    if covdata_out
        % The saved data of covariance functions; last cell is K (computed below)
        covdata = cell(length(cov)+1,1);
    end    
    for ii = 1:length(cov)                      % iteration over summand functions
        f = cov(ii); if iscell(f{:}), f = f{:}; end % expand cell array if necessary
        if covdata_out
            [Kii, covdata{ii}] = feval(f{:}, hyp(v==ii), x, z);
        else
            Kii = feval(f{:}, hyp(v==ii), x, z);
        end
        K = K + Kii;              % accumulate covariances
    end
    if covdata_out
        % Save cov data
        covdata{end} = K;
    end
else                                                               % derivatives
    if i<=length(v)
        if covdata_out
            assert(has_covdata, 'Covariance function data must be provided.');
            covdata = varargin{1};
        end
        vi = v(i);                                       % which covariance function
        j = sum(v(1:i)==vi);                    % which parameter in that covariance
        f  = cov(vi);
        if iscell(f{:}), f = f{:}; end         % dereference cell array if necessary
        
        % compute derivative        
        if has_covdata
            if covdata_out
                [K, covdata{vi}] = feval(f{:}, hyp(v==vi), x, z, j, varargin{1}{vi});
            else
                K = feval(f{:}, hyp(v==vi), x, z, j, varargin{1}{vi});                
            end
        else
            % covdata_out must be false because of assert() above
            K = feval(f{:}, hyp(v==vi), x, z, j);
        end
    else
        error('Unknown hyperparameter')
    end
end