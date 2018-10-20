function [amp,phase]=isSameFreqSinusoid(f,system,fig)
    N=64;t = (1:N)';
    %% generate input with desired frequency and random phase
    input = cos(f*t+randi(N,1)*2*pi/N)+1;
    %% compute output
    output = system(input);
    %% generate basis of the subspace
    X = [ones(N,1),cos(f*t),sin(f*t)];
    %% linear regression
    b = (X'*X)\X'*output; 
    %% examine if regression is complete
    result = all(abs(X*b-output)<1e-10);
    if result 
        disp('output is in the same subspace');
    else
        disp('output is outside the input subspace');
    end
    %% calculate amplitude and phase shift
    amp = (max(output) - min(output))/2;
    output_ = output-mean(output);  % substract mean
    output_n= output_/amp; % normalize
    phase = acos( dot(input-1,output_n)/norm(input)/norm(output) );
    %% plot
    if fig
        figure; hold on
        plot(input)
        plot(output_n+1)
        legend('input','normalized output')
        xlabel('time')
    end
    
    
    