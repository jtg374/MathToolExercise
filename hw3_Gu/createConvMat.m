function X=createConvMat(x,M)
    % size(x) should be [N,1]
    xPad = [x;zeros(M-1,1)];
    for ii=1:M
        X(:,ii) = circshift(xPad,M-1);
    end