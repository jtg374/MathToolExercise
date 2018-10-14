close all; clear
%% 1 LSI system characterization
%%
% * a) check linearity and shift-invariance
%%
N=64;
I = eye(64); % generate a series of impulse input
%%
% System 1 is not shift-invariant. The reponse gain grows with time. 
figure; hold on
positions = randi(N,1,4);
for p=positions
    s = num2str(p);
    eval(...
        ['plot(unknownSystem1(I(:,',s,')),''DisplayName'',''impulse at position ',s,''')'])
end
xlabel('time')
ylabel('response')
title('Impulse response of system 1 at different positions')
legend()
%%
% System 1 is not linear
output1 = unknownSystem1(I(:,positions(1)));
output2 = unknownSystem1(I(:,positions(2)));
sumOfOutput = output1+output2;
sumOfInput = I(:,positions(1))+I(:,positions(2));
all(sumOfOutput == unknownSystem1(sumOfInput))
%%
% System 2 is shift-invariant
figure; hold on
positions = randi(N,1,4);
for p=positions
    s = num2str(p);
    eval(...
        ['plot(unknownSystem2(I(:,',s,')),''DisplayName'',''impulse at position ',s,''')'])
end
xlabel('time')
ylabel('response')
title('Impulse response of system 2 at different positions')
legend()
%%
% System 2 is not linear because of a nonzero offset
output1 = unknownSystem2(I(:,positions(1)));
output2 = unknownSystem2(I(:,positions(2)));
sumOfOutput = output1+output2;
sumOfInput = I(:,positions(1))+I(:,positions(2));
all(sumOfOutput == unknownSystem2(sumOfInput))
%%
figure; hold on
plot(sumOfOutput)
plot(unknownSystem2(sumOfInput))
xlabel('time');ylabel('response')
legend('sum on output side','sum on input side')
title('nonlinearity of system 2')
%%
% System 3 is shift-invariant
figure; hold on
positions = randi(N,1,4);
for p=positions
    s = num2str(p);
    eval(...
        ['plot(unknownSystem3(I(:,',s,')),''DisplayName'',''impulse at position ',s,''')'])
end
xlabel('time')
ylabel('response')
title('Impulse response of system 3 at different positions')
legend()
%%
% System 3 is linear
% positions = randi(N,1,4);
output1 = unknownSystem3(I(:,positions(1)));
output2 = unknownSystem3(I(:,positions(2)));
sumOfOutput = output1+output2;
sumOfInput = I(:,positions(1))+I(:,positions(2));
all(sumOfOutput == unknownSystem3(sumOfInput))

%%
% * b) response to sinusoid
freqs = [1 2 4 8]*2*pi/N;
t = (1:N)';
f = freqs(1);
input = cos(f*t+rand()*2*pi)+1;
output = unknownSystem1(input);
output_ = output-mean(output);
output_n = output_/max(abs(output_));
figure;plot(acosd(output_n))
%%
figure;hold on
plot(input);
plot(output)


%%
close all