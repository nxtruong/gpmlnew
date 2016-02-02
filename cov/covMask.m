function varargout = covMask(cov, hyp, x, z, i, varargin)
% Apply a covariance function to a subset of the dimensions only. The subset can
% either be specified by a 0/1 mask by a boolean mask or by an index set.
%
% This function doesn't actually compute very much on its own, it merely does
% some bookkeeping, and calls another covariance function to do the actual work.
%
% The function was suggested by Iain Murray, 2010-02-18 and is based on an
% earlier implementation of his dating back to 2009-06-16.
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2012-11-17.
% Modified and copyright (c) by Truong X. Nghiem, 2016-01-28.
%
% See also COVFUNCTIONS.M.

mask = fix(cov{1}(:));                    % either a binary mask or an index set
cov = cov(2);                                 % covariance function to be masked
if iscell(cov{:}), cov = cov{:}; end        % properly unwrap nested cell arrays
nh_string = feval(cov{:});    % number of hyperparameters of the full covariance

if max(mask)<2 && length(mask)>1, mask = find(mask); end    % convert 1/0->index
D = length(mask);                                             % masked dimension
if nargin<3, varargout = {num2str(eval(nh_string))}; return, end    % number of parameters
if nargin<4, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode

if eval(nh_string)~=length(hyp)                          % check hyperparameters
    error('number of hyperparameters does not match size of masked data')
end

varargout = cell(1, nargout);

if nargin<5 || isempty(i)                                          % covariances
    if dg
        [varargout{:}] = feval(cov{:}, hyp, x(:,mask), 'diag', [], varargin{:});
    else
        if xeqz
            [varargout{:}] = feval(cov{:}, hyp, x(:,mask), [], [], varargin{:});
        else
            [varargout{:}] = feval(cov{:}, hyp, x(:,mask), z(:,mask), [], varargin{:});
        end
    end
else                                                               % derivatives
    if i <= eval(nh_string)
        if dg
            [varargout{:}] = feval(cov{:}, hyp, x(:,mask), 'diag', i, varargin{:});
        else
            if xeqz
                [varargout{:}] = feval(cov{:}, hyp, x(:,mask), [], i, varargin{:});
            else
                [varargout{:}] = feval(cov{:}, hyp, x(:,mask), z(:,mask), i, varargin{:});
            end
        end
    else
        error('Unknown hyperparameter');
    end
end