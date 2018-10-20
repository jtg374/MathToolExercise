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
%%
% <include>isSameFreqSinusoid.m</include>
%%

for sys = 1:3
    eval(sprintf('system = @unknownSystem%d',sys));
    disp system
    for f = [1,2,4,8]
        disp(['frequency: ',num2str(f),'*2*pi/64'])
        [amplitude, phaseShift] = isSameFreqSinusoid(f*2*pi/N,system,0)
    end
end
%%        
% Only System 3 project sinusoids into sinuisoid of the same frequency. 
isSameFreqSinusoid(2*2*pi/N,@unknownSystem3,1)
%%
% * c)
%%
% System 1 does not have an unique impulse response. 
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
xlabel(sprintf('freq (2?/%d rad/sample)',N))
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
xlabel(sprintf('freq (2?/%d rad/sample)',N))
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
% Period is 32/8192Hz = 2^-8 = 0.039 seconds. freqency is 2^8=256 Hz, closest to
% middle C (C4, 261.6Hz)
%% * b)
sigF = fft(sig);
figure
bar((-N/2):(N/2-1),fftshift(abs(sigF)))
ylabel('amplitude')
ticks=-(N/2):256:(N/2-1);
xticks(ticks)
xticklabels(ticks/N*8192)
xlabel(sprintf('freq (Hz)'))

%%
% There are huge gaps between bars, i.e. sinusoids of many frequencies would
% give zero response, unless the frequency is a multiple of 256 Hz. 
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
% The spectrum indeed peaks every time the frequency is a multiple of 1024/3 Hz
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
% So it still has a period of 2^-8 second, or 32 samples. 
%%
figure; hold on
nT = 1:32;
plot(nT/8192,sig(nT));
plot(nT/8192,sigG(nT));
legend('f(n)','g(n')
xlabel('time (s)')
%%
% the wave looks more smooth, as in the Dourier spectrum there is no high
% frequency components, but is only a linear combination of 256 Hz and 512
% Hz sinusoids and a DC shift. 
%%
sound(sigG)
%%
sound(sig)
%%
% The timbre of g(n) is brighter. 
%% 3 Gabor filter
%%
% * a)
%%
n = -12:12;
sigma=3.5;
omega=10*2*pi/64;
gabor=exp(-(n/sigma).^2/2).*cos(omega*n);
figure
plot(n,gabor);title('gabor filter')
%%
gaborF=fft(gabor,64);
figure
plot(-32:31,fftshift(abs(gaborF)))
%%
% This is a bandpass filter that selectively filter frequency of 10. ?
% determines the bandpass frequency and 1/? determines the width of the band.
gaborWiderAt12 = exp(-(n).^2/2).*cos(omega*1.2*n);
gaborWiderAt12F=fft(gaborWiderAt12,64);
figure
plot(-32:31,fftshift(abs(gaborWiderAt12F)))

%%
% * b )
%%
% f = 10*2?/64 will give the largest response of all sinusoid cos(fn). 
%%
% because the convolution at position 0 is
%%
% $$\sum_{n=-12}^12 gabor(-n)cos(fn)$$
%%
% is the real part of the f*64/2?'th Fourier coefficient. and since gabor
% is evenly symmetric, thus the imagionary parts are all zero, the real
% part is just the amplitude. 
%% 
% the max amplitude is at the position
[~,ind] = max(abs(gaborF));
ind-1
%%
% By inspection, the t
%%
% * c)
%% 4 Deconvolution of the Hawmodynamic Response
%%
%%
load hrfDeconv.mat;
figure;hold on
plot(r);stem(x);
legend('r','x')
xlabel('time (s)')
%%
% <include>createConvMat.m</include>
%%
M = 15;
X = createConvMat(x,M);
%%
% Verify the matrix
for ii = 1:5
    disp(['test filter ',num2str(ii)])
    h = rand(M,1);
    if all(X*h == conv(x,h)); disp('X*h == conv(x,h)'); else; disp('Fail'); end
end
%%
% Verified.
%%
figure;imagesc(X);colorbar;title('conv matrix')
%%
% a step-like structure
%%
% * b)
%%
h_opt = (X'*X)\X'*r;
figure; plot(h_opt); xlabel('time (s)'); legend('h')
grid on
%%
%
%%
r_est = conv(x,h_opt);
figure; hold on
plot(r);plot(r_est);
legend('r','r_{est}')
xlabel('time (s)')
%%
% * c)
%%
M=15;
hF = fft(h);
hP = hF.*conj(hF);
figure
freqs = (1:M) - ceil(M/2);
freqs = freqs/M;
plot(freqs,fftshift(hP))
xlabel('freq (Hz)');ylabel('power')
title('power spectrum of h')
%%
% It's a low pass filter, only frequencies lower than 0.05 Hz are passed. 
%%
close all