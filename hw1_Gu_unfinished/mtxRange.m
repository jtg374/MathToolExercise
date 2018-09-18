function v = mtxRange(M)
    [U, S, ~] = svd(M);
    SS = diag(S);
    UR = U(:,SS>1e-10);
    n = size(UR,2);
    if n
        v = UR*rand(n,1);
    else
        v = false;
        disp('Range Space is empty')
    end