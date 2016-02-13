% Evaluate the order R elementary symmetric polynomial Newton's identity aka
% the Newton-Girard formulae: http://en.wikipedia.org/wiki/Newton's_identities
%
% Fast version modified by Truong X. Nghiem.
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-01-10.
% Copyright (c) by Truong X. Nghiem, 2016-01-20.

function E = elsympol(Z,R)
% evaluate 'power sums' of the individual terms in Z
sz = size(Z);
E = zeros([sz(1:2),R+1]);                   % E(:,:,r+1) yields polynomial r
E(:,:,1) = ones(sz(1:2)); if R==0, return, end  % init recursion

P = repmat(sum(Z,3),1,1,R);
E(:,:,2) = P(:,:,1);      if R==1, return, end  % init recursion

Zc = Z;
for r=2:R
    Zc = Zc.*Z;
    P(:,:,r) = sum(Zc,3);
    % Compute E for i = 1 is simpler than for any other i
    E(:,:,r+1) = P(:,:,1).*E(:,:,r)/r;
    for i=2:r
        E(:,:,r+1) = E(:,:,r+1) + P(:,:,i).*E(:,:,r+1-i)*((-1)^(i-1)/r);
    end
end
end