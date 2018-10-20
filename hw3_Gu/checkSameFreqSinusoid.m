function phaseDiff=checkSameFreqSinusoid(f,p,system,fig)
    t = (1:64)';
    input = cos(f*t+p)+1;
    output = system(input);
    output_ = output-mean(output); % substract mean
    output_n = output_/max(abs(output_)); % and normalize
    if fig
        figure; hold on
        plot(input)
        plot(output_n+1)
        legend('input','normalized output')
        xlabel('time')
    end
    phaseDiff = acos(dot(input,output)/(norm(input)*norm(output)));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    