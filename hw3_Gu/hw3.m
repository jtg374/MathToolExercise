close all; clear
%% 1 LSI system characterization
%%
% * a) check linearity and shift-invariance
%%
N=64;
I = eye(64); % generate a series of impulse input
%%
% System 1 is not shift-invariant. The reponse gain grows with time. 
figure; hold on; grid on
positions = randperm(N,4);
for p=positions
    output = unknownSystem1(I(:,p));
    line = plot(output);
    set(line,'Displayname',['impulse response at ',num2str(p)]);
end
xlabel('time')
ylabel('response')
title('Impulse response of system 1 at different positions')
legend('Location','southeast')
%%
% System 1 is nonlinear
output1 = unknownSystem1(I(:,positions(1)));
output2 = unknownSystem1(I(:,positions(2)));
sumOfOutput = output1+output2;
sumOfInput = I(:,positions(1))+I(:,positions(2));
all(sumOfOutput == unknownSystem1(sumOfInput))
%%
% System 2 is shift-invariant
figure; hold on; grid on
positions = randperm(N,4);
for p=positions
    output = unknownSystem2(I(:,p));
    line = plot(output);
    set(line,'Displayname',['impulse response at ',num2str(p)]);
end
xlabel('time')
ylabel('response')
title('Impulse response of system 2 at different positions')
legend('Location','southeast')
%%
% System 2 is nonlinear because of a nonzero offset
output1 = unknownSystem2(I(:,positions(1)));
output2 = unknownSystem2(I(:,positions(2)));
sumOfOutput = output1+output2;
sumOfInput = I(:,positions(1))+I(:,positions(2));
all(sumOfOutput == unknownSystem2(sumOfInput))
%%
figure; hold on; grid on
plot(sumOfOutput)
plot(unknownSystem2(sumOfInput))
xlabel('time');ylabel('response')
legend('sum on output side','sum on input side','Location','southeast')
title('nonlinearity of system 2')
%%
% System 3 is shift-invariant
figure; hold on; grid on
positions = randperm(N,4);
for p=positions
    output = unknownSystem3(I(:,p));
    line = plot(output);
    set(line,'Displayname',['impulse response at ',num2str(p)]);
end
xlabel('time')
ylabel('response')
title('Impulse response of system 3 at different positions')
legend('Location','southeast')
%%
% System 3 is possibly linear
positions = randi(N,1,4);
g1=1;g2=1;
output1 = unknownSystem3(I(:,positions(1)));
output2 = unknownSystem3(I(:,positions(2)));
sumOfOutput = g1*output1+g2*output2;
sumOfInput = g1*I(:,positions(1))+g2*I(:,positions(2));
all(sumOfOutput == unknownSystem3(sumOfInput))

%%
% * b) response to sinusoid
%%
% <include>isSameFreqSinusoid.m</include>
%%

for sys = 1:2
    eval(sprintf('system = @unknownSystem%d',sys));
    disp system
    for f = [1,2,4,8]
        disp(['frequency: ',num2str(f),'*2*pi/64'])
        isSameFreqSinusoid(f*2*pi/N,system);
    end
end
%%        
% Only System 3 project sinusoids into sinuisoid of the same frequency. 
system = @unknownSystem3
for f = [1,2,4,8]
    disp(['frequency: ',num2str(f),'*2*pi/64'])
    [amplitude,phaseShift]=isSameFreqSinusoid(f*2*pi/N,@unknownSystem3)
end    
    
%%
% This is predicted by the Fourier transfrom of system's impulse response
impulseResponse = unknownSystem3(I(:,1));
impulseResponseDFT = fft(impulseResponse);
ampF = abs(impulseResponseDFT); phaseF = angle(impulseResponseDFT);
freqs = [1,2,4,8];
figure
ax1=subplot(2,1,1); hold on; grid on
bar((-N/2):(N/2-1),fftshift(ampF))
bar(freqs,ampF(freqs+1))
ylabel('amplitude')
ax2=subplot(2,1,2); hold on; grid on
bar((-N/2):(N/2-1),fftshift(phaseF))
bar(freqs,phaseF(freqs+1))
ylabel('phase (rad)')
linkaxes([ax1 ax2],'x')
xlabel(sprintf('freq (2?/%d rad/sample)',N))
amplitude = ampF(freqs+1)
phaseShift = phaseF(freqs+1)
%%
% * c)
%%
% System 1 is neither linear nor shift-invariant, and it doesn't project
% siusoids into the sinusoid subspaces with same frequency. 
%%
% System 2 is nonlinear but shift-invariant, and it doesn't project
% siusoids into the sinusoid subspaces with same frequency. 
%%
% System 3 seems both linear and shift-invariant, and it projects
% siusoids into the sinusoid subspaces with same frequency. 
%%
% Linearity and shift-invariance and indepedent properties so that they can
% be tested seperately, but only if a system is both linear and
% shift-invariant, its response to sinuisoid would be a sinusoid of the
% same frequency and the output gain and phase shift relative to input can
% be predicted by the Fourier transform of the system's impulse reponse. So
% if a system always output a sinusoid given any sinusoid input, with a
% fixed gain and phase shift that do not vary with input strength and
% phase, then this system is guarenteed to be a LSI system. 
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
title('Fourier amplitude spectrum of gabor')
xlabel('spatial frequency (1/64 cycle/sample)')
%%
% This is a bandpass filter that selectively filter frequency of 10*2?/64
% samples.
% It looks like two gaussian functions centered at 10 and -10.
% This shape is inherited from the gaussian function and the sinusoid. 
% ? determines the bandpass frequency and 1/? determines the width
% of the band.
%% 
% If I let ?=12**2?/64 and ?=1
gaborWiderAt12 = exp(-(n).^2/2).*cos(omega*1.2*n);
gaborWiderAt12F=fft(gaborWiderAt12,64);
figure
plot(-32:31,fftshift(abs(gaborWiderAt12F)))

%%
% * b )
%%
% f = 10*2?/64 will give the largest response among all sinusoid cos(fn). 
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
[Fmax,ind] = max(abs(gaborF));
ind-1
%%
% The period of this sinusoid is 2?/f samples
64/10
%%
% By eye inspection, this is roughly the distance between peaks in the
% filter itself. 
figure; hold on
plot(n,gabor);title('gabor filter')
plot([6.4,6.4],[-1 1],'b--')
plot([6.4,6.4]*2,[-1 1],'b--')
plot([0,0],[-1 1],'b--')
plot([6.4,6.4]*-1,[-1 1],'b--')
plot([6.4,6.4]*-2,[-1 1],'b--')
xticks(-12.8:6.4:12.8)
%% 
% By eye inspection, sinusoids of frequency 5/64 cycle/sample and 15/64
% cycle/sample will give about 25% of the maximal amplitude
%%
gaborF=fft(gabor,64);
figure; hold on
plot(-32:31,fftshift(abs(gaborF)))
plot([-32,31],[Fmax Fmax]*0.25,'b--')
plot([5 5],[0 Fmax],'b--')
plot([15 15],[0 Fmax],'b--')
xticks(-35:10:35)
title('Fourier amplitude spectrum of gabor')
xlabel('spatial frequency (1/64 cycle/sample)')
%%
% * c)
%%
n=1:64;
input5  = cos( 5*2*pi/64*n);
input10 = cos(10*2*pi/64*n);
input15 = cos(15*2*pi/64*n);
rLow = conv(input5 ,gabor); 
rMed = conv(input10,gabor);
rHih = conv(input15,gabor);
ampLow = abs(fft(rLow,64)); 
display(['low freq response amplitude: ',num2str(ampLow(5+1))])
ampMed = abs(fft(rMed,64));
display(['medium freq response amplitude: ',num2str(ampMed(10+1))])
ampHih = abs(fft(rHih,64)); 
display(['high freq response amplitude: ',num2str(ampHih(15+1))])
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
    if all(X*h == conv(x,h)); disp('X*h is equal to conv(x,h)'); else; disp('Fail'); end
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
% HRF rises to peak in 5 seconds and drops into negative then recover. The
% positive peak is about 3 times the negative valley. The curve looks like
% a 3/4 cycle sine wave multiplied by a decay. It takes about 15 second
% for HRF to recover to baseline. 
%%
% The estimated HRF fits the response well. 
r_est = conv(x,h_opt);
figure; hold on
plot(r);plot(r_est);
legend('r','r_{est}')
xlabel('time (s)')
%%
% * c)
%%
M=15;
hF = fft(h,M);
hP = hF.*conj(hF);
figure
freqs = (0:M-1) - floor(M/2);
freqs = freqs/M;
plot(freqs,fftshift(hP))
xlabel('freq (Hz)');ylabel('power')
xticks(-.4:.2:.4)
title('power spectrum of h')
%%
% It's a low pass filter, only frequencies lower than 0.05 Hz are passed. 
%%
close all