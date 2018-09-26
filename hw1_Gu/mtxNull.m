function v = mtxNull(M)
    [~, S, V] = svd(M);
    [m,n] = size(M);
    SS = diag(S);
    VN = V(:,abs(SS)<1e-10);
    if m<n
        VN = [VN V(:,(m+1):n)];
    end
    nN = size(VN,2);
    if nN
        v = VN*rand(nN,1);
    else
        v = false;
        disp('Null Space is empty')
    end