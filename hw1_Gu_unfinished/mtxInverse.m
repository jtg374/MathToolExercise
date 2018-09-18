function invM = mtxInverse(M)
    [U,S,V] = svd(M);
    invS = zeros(size(S'))
    for ii = 1:min(size(S))
        s_i = SS(ii,ii);
        if s_i~=0
            invS(ii,ii) = 1/s_i;
        end
    end
    invM = V*invS*U;


