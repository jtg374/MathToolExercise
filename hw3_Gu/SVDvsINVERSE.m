X = rand(3,2);
[U S V] = svd(X);
Si = zeros(size(S'));
for ii = 1:min(size(X))
    Si(ii,ii) = 1./S(ii,ii);
end
direct = X'*X\X'
% direct_obsolete = inv(X'*X)*X'
indirect = V*S'*S\S'*U'
indirect_tedious = V*Si*U'
% indirect_obsolete = V*inv(S'*S)*S'*U'
