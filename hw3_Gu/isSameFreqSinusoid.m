function [amp,phase]=isSameFreqSinusoid(f,system)
    N=64;t = (1:N)';
    %% generate input with desired frequency and random phase
    inputPhase = randi(N,1)*2*pi/N;
    input = cos(f*t-inputPhase)+1;
    %% compute output
    output = system(input);
    %% generate orthonormal basis of the subspace
    X = [ones(N,1),cos(f*t),sin(f*t)];
    %% linear regression
    b = X'*X\X'*output;
    %% examine if regression is complete
    result = all(abs(X*b-output)<1e-10);
    if result 
        disp('output is in the same subspace');
    else
        disp('output is outside the input subspace');
    end
    %% calculate amplitude and phase shift
    amp = sqrt(b(2)^2+b(3)^2);
    phase = atan2(b(3),b(2)) - inputPhase;
    phase = mod(phase+pi,2*pi)-pi; % so that -pi<=phase<pi
    
    