function v = mtxRange(M)
    [U, S, ~] = svd(M);
    SS = diag(S);
    UR = U(:,SS~=0);
    n = size(UR,2);
    if n
        v = UR*rand(n,1);
    else
        v = inf;
        disp('Range Space is empty')
    end