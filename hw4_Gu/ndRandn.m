function samples = ndRandn(mean,cov,num)
    if nargin<3
        num = 1;
    end
    N = length(mean);
    if size(cov,1) ~= N || size(cov,2) ~=N
        disp('ERROR: covariance matrix size doesn''t match mean vector')
    end
    [U,S,~] = svd(cov);
    cov_sqrt = U*sqrt(S);
    samples = mean + cov_sqrt * randn([length(mean),num]);