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
% * c)
%%
% Fourier transform of System 2's impulse response
impulseResponse = unknownSystem2(I(:,1));
impulseResponseDFT = fft(impulseResponse);
figure;
ax1=subplot(2,1,1);
bar((-N/2):(N/2-1),fftshift(abs(impulseResponseDFT)))
ylabel('amplitude')
ax2=subplot(2,1,2);
bar((-N/2):(N/2-1),fftshift(angle(impulseResponseDFT)))
ylabel('phase (rad)')
linkaxes([ax1 ax2],'x');
xlabel(sprintf('freq (2?/%d)',N))
%%
% Fourier transform of System 3's impulse response
impulseResponse = unknownSystem3(I(:,1));
impulseResponseDFT = fft(impulseResponse);
figure
ax1=subplot(2,1,1);
bar((-N/2):(N/2-1),fftshift(abs(impulseResponseDFT)))
ylabel('amplitude')
ax2=subplot(2,1,2);
bar((-N/2):(N/2-1),fftshift(angle(impulseResponseDFT)))
ylabel('phase (rad)')
linkaxes([ax1 ax2],'x')
xlabel(sprintf('freq (2?/%d)',N))
%% 2 Fourier transform of periodic signals
%%
% * a)
%%
N=2048;
n = 1:N;
sig = mod(n,32)/32; % generate a sawtooth
figure;plot((1:N)/8192,sig);xlabel('time (s)');
xlim([0 N/8192]);title('sawtooth signal')
sound(sig,8192)
%%
% Duration is 2048/8192Hz = 0.25 seconds. 
%%
% Period is 32/8192Hz = 2^-8 seconds. freqency is 2^8=256 Hz, closest to
% middle C (C4, 261.6Hz)
%% * b)
sigF = fft(sig);
figure
bar((-N/2):(N/2-1),fftshift(abs(sigF)))
ylabel('amplitude')
ticks=-1024:256:1023;
xticks(ticks)
xticklabels(ticks/N*8192)
xlabel(sprintf('freq (Hz)'))

%%
% There are huge gaps between bars, i.e. sinusoids of many frequencies would
% give zero response, unless the frequency is an integral multiple of
% 256 Hz. 
%%
% Generally, the Fourier amplitude spectrum which has periodic peak
% pattern indicates that in time domain, the signal is periodic with the
% same period. 
sig24 = mod(n,24)/24; % generate a new sawtooth of 24-sample period 
% the frequency of this signal is 8192/24 = 1024/3 Hz
sig24F = fft(sig24);
figure
bar((-N/2):(N/2-1),fftshift(abs(sig24F)))
ylabel('amplitude')
ticks=-1024:(2048/24*3):1023;
xticks(ticks)
xticklabels(ticks/N*8192)
xlabel(sprintf('freq (Hz)'))
%%
% The spectrum indeed peaks at where the frequency is an integral multiple
% of 1024/3 Hz
%% * c)
N=2048;
sigG = (1+cos(n*2*pi*64/N)).^2;
sigGF = fft(sigG);
figure
bar((-N/2):(N/2-1),fftshift(abs(sigGF)))
ylabel('amplitude')
ticks=-1024:256:1023;
xticks(ticks)
xticklabels(ticks/N*8192)
xlabel(sprintf('freq (Hz)'))
%%
% This spectrum only has nonzero at frequency 0, 256 and 512 Hz. 
%%
% 

%%
close all