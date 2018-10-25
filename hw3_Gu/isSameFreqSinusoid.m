function [amp,phaseDiff]=isSameFreqSinusoid(f,system)
    % The following equation test whether the output is of the same
    % frequency with the input. The test uses linear regression, which is
    % equivalent with calculating the Fourier coefficients of the output
    % corresponding to the input frequency term and the DC term, and check
    % if the regression is complete.
    N=64;t = (0:(N-1))';
    %% generate input with desired frequency and random phase
    inputPhase = randi(N,1)*2*pi/N;
    input = cos(f*t + inputPhase)+1;
    %% compute output
    output = system(input);
    %% generate orthonormal basis of the subspace
    X = [ones(N,1),cos(f*t),-sin(f*t)]; % -sin because of conjugate in complex inner products
    %% linear regression
    % The SVD method gives a slightly different pseudoinverse matrix and
    % the error is at the same order as using the backslash method in
    % MATLAB. I use backslash instead for concisement
    b = (X'*X)\X'*output; % standard linear regression operation (projection of r to X's subspace, analogous to vector projection)
%     [U,S,V] = svd(X);
%     Si = zeros(size(S'));
%     for ii = 1:min(size(X))
%         Si(ii,ii) = 1./S(ii,ii);
%     end
%     b = V*Si*U'*output;
%     X*b-output
    %% examine if regression is complete
    result = all(abs(X*b-output)<1e-10); % to avoid floating number error, equivalent to using round and ==
    if result 
        disp('output is in the same subspace');
    else
        disp('output is outside the input subspace');
    end
    %% calculate amplitude and phase shift
    amp = sqrt(b(2)^2+b(3)^2);
    phase = atan2(b(3),b(2));
    phaseDiff = phase - inputPhase;
    phaseDiff = mod(phaseDiff+pi,2*pi)-pi; % so that -pi<=phaseDiff<pi
    
    