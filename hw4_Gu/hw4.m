close all; clear;
%% 1 Bayes rule and eye color
%%
% Let blue eye gene = b, brown eye gene = B. given Father has blue eyes, he
% must be
%%
% 
% $$ P(Father = bb) = 1 $$
% 
%%
% given mother has brown eyes, She must carry a B. we assume
%%
% 
% $$ P(Mother = BB) = P(Mother = Bb) = 1/2 $$
% 
%%
% because we have no knowledge about the other allele. 
%%
% * a)
%%
% 
% $$ P(mother = Bb | First Child = brown) $$
%
%%
% 
% $$ = P(First Child = brown | mother = Bb ) P(mother = Bb) / P(First Child = brown) $$
%
%%
%
% $$ = P(First Child = brown | mother = Bb ) * 1/2 / P(First Child = brown) 
%
%%
% because Children must carry a b from Father. 
%%
% 
% $$ P(First Child = brown | Mother = Bb) = P(First Child = Bb | Father = bb, Mother = Bb) = 1/2 $$
%
%%
% 
% $$ P(First Child = brown) = P(First Child = Bb | Father = bb) $$
%
%%
% 
% $$ = P(First Child = Bb | Mother = Bb, Father = bb) * P(Mother = Bb | Father = bb) $$
% 
%%
% 
% $$ + P(First Child = Bb | Mother = BB, Father = bb) * P(Mother = BB | Father = bb) $$
% 
%%
% $$ = 1/2 * 1/2 + 1 * 1/2 = 3/4 $$
%%
% 
% $$ P(mother = Bb | First Child = brown) $$
% 
%%
% 
% $$ = 1/2 * 1/2 / (3/4)  = 1/3 $$
% 
%%
% * b)
%%
% 
% $$ P(mother = Bb | Second Child = brown, First Child = brown) $$
%
%%
% 
% $$ = P(mother = Bb | Second Child = Bb, First Child = Bb, Father = bb) $$
%
%%
% 
% $$ = P(Second Child = Bb | mother = Bb, First Child = Bb, Father = bb ) P(mother = Bb) / P(Second Child = Bb | First Child = Bb, Father = bb) $$
%
%%
% 
% $$ = P(Second Child = Bb | mother = Bb, Father = bb ) P(mother = Bb) / P(Second Child = Bb | First Child = Bb, Father = bb) $$
%
%%
%
% $$ = 1/2 * 1/3 / P(Second Child = Bb | First Child = Bb, Father = bb) $$
%
%%
% 
% $$ P(Second Child = Bb | First Child = Bb, Father = bb)  $$
%
%%
% 
% $$ = P(Second Child = Bb | Mother = Bb, First Child = Bb, Father = bb) * P(Mother = Bb | First Child = Bb, Father = bb) $$
% 
%%
% 
% $$ + P(Second Child = Bb | Mother = BB, First Child = Bb, Father = bb) * P(Mother = BB | First Child = Bb, Father = bb) $$
% 
%%
% 
% $$ = P(Second Child = Bb | Mother = Bb, Father = bb) * P(Mother = Bb | First Child = Bb, Father = bb) $$
% 
%%
% 
% $$ + P(Second Child = Bb | Mother = BB, Father = bb) * (1 - P(Mother = Bb | First Child = Bb, Father = bb) ) $$
% 
%%
% $$ = 1/2 * 1/3 + 1 * 2/3 = 5/6
%%
% 
% $$ P(mother = Bb | First Child = brown, second Child = brown) $$
% 
%%
% 
% $$ = 1/2 * 1/3 / (5/3)  = 1/5 $$
% 
%%
% * c)
%%
% let 
%%
%
% $$ P_{N} = P(mother = Bb | N children = Bb, Father = bb) $$
%
%%
%
% $$ = P(mother = Bb | N-1 children = Bb, Father = bb) P(Nth child = Bb | Mother = Bb, Father = bb) / P(Nth child = Bb | N-1 children = Bb, Father = bb) $$
%
%%
%
% $$ = P_{N-1} * 1/2 / P(Nth child = Bb | N-1 children = Bb, Father = bb) $$
%
%%
% 
% $$ P(Nth child = Bb | N-1 children = Bb, Father = bb) $$
% 
%%
%
% $$ = P(Nth child = Bb | Mother = Bb, Father = bb) * P( Mother = Bb | N-1 children = Bb, Father = bb) $$
%
%%
%
% $$ + P(Nth child = Bb | Mother = BB, Father = bb) * P( Mother = BB | N-1 children = Bb, Father = bb) $$
%
%%
%
% $$ = 1/2 P_{N-1} + (1 - P_{N-1}) = 1 - 1/2 P_{N-1} $$
%
%%
%
% $$ P_{N} = 1/2 P_{N-1} / (1 - 1/2 P_{N-1}) $$
%
%%
% $$ 1/P_{N} = 2/P_{N-1} - 1
%%
% $$ 1/P_{N} - 1 = 2 ( 1/P_{N-1} - 1 )
%%
% $$ p_{N} = 1/(1+(1/P_1 - 1)*2^(N-1)) $$
%% 
% $$ p_1 = 1/3 $$
%%
% $$ p_{N} = 1/(1+2^N)

%% 2 Poisson neurons
%% 
% * a)
%%
k = 0:20;
mu = 5; % mean rate over the interval
p = (mu.^k * exp(-mu))./factorial(k); % PDF for infinite length
p = p/sum(p); % normalize
%%
% <include>randp.m</include>
%%
h = zeros(21,4); % allocate space for histogram
for m=2:5
    N=10^m; % sample size
    samples = randp(p,N); % generate N samples
    h(:,m-1) = hist(samples,k)/N;
end
figure; hold on
line=plot(k,p);
handle = bar(k,h);
label = cell(1,4);
label{1} = '10^2 samples'; label{2} = '10^3 samples'; label{3} = '10^4 samples'; label{4} = '10^5 samples';
legend([line,handle],['PDF',label]);
xlabel('spike counts')
ylabel('freq')
title('spike count distribution for different sample number')
%%
% As we can see, as sample number increase, distribution becomes closer to
% p
%%
for m = 2:5
    SE(m-1) = sum((h(:,m-1)-p').^2);
end
figure; plot(2:5,SE,'b-*')
xlabel('log(#sample)')
ylabel('squared difference')
%%
% * b)
mu1 = 2; % rate of the second neuron
q = (mu1.^k * exp(-mu1))./factorial(k);
q = q/sum(q); % PDF of the spike count of neuron 2
%%
pq = conv(p,q)'; % PDF of the sum of spikes
h = zeros(41,4);
for m=2:5
    N=10^m;
    samples = randp(pq,N);
    h(:,m-1) = hist(samples,0:40)/N;
end
figure; hold on
line = plot(0:40,pq);
handle = bar(0:40,h);
label = cell(1,4);
label{1} = '10^2 samples'; label{2} = '10^3 samples'; label{3} = '10^4 samples'; label{4} = '10^5 samples';
xlabel('sum of spike count')
ylabel('freq')
legend([line,handle],['PDF',label]);
title('spike count summation distribution for different sample number')
%%
% Again, as sample size increase, the distribution becomes closer to the
% PDF
%%
for m = 2:5
    SE(m-1) = sum((h(:,m-1)-pq).^2);
end
figure; plot(2:5,SE,'b-*')
xlabel('log(#sample)')
ylabel('squared difference')
%%
% * c)
%%
k = 0:40;
mu2 = 7; 
r = (mu2.^k * exp(-mu2))./factorial(k);
r = r/sum(r);
if all(abs(r'-pq)<1e-5)
    disp('two distributions are the same')
else
    disp('two distributions are different')
end
%%
% If we record a new spike train I can't tell whether the spikes came from
% one or two neurons just by looking at their distribution of spike counts.
%% 3 Central Limit Theorem
%%
% * a)
h = zeros(21,4 );
for m=2:5
    N=10^m;
    samples = rand(N,2);
    samples = mean(samples,2);
    h(:,m-1) = hist(samples,0:0.05:1)/N;
end
figure; hold on
handle = bar(0:0.05:1,h);
label = cell(1,4);
label{1} = '10^2 sets'; label{2} = '10^3 sets'; label{3} = '10^4 sets'; label{4} = '10^5 sets';
legend(handle,label)
title('distibution of the average of 2 samples')
%%
% As the sample size increase, the distribution becomes more like a
% triangular shape. This is the result of convoluting two uniform
% distributions (triangles) and scale the x axis by 2
%%
% * b)
%%
h = zeros(21,4);
for m=2:5
    N=10^m;
    samples = rand(N,3);
    samples = mean(samples,2);
    h(:,m-1) = hist(samples,0:0.05:1)/N;
end
figure; hold on
handle = bar(0:0.05:1,h);
label = cell(1,4);
label{1} = '10^2 sets'; label{2} = '10^3 sets'; label{3} = '10^4 sets'; label{4} = '10^5 sets';
legend(handle,label)
title('distibution of the average of 3 samples')
%%
h = zeros(21,4 );
for m=2:5
    N=10^m;
    samples = rand(N,4);
    samples = mean(samples,2);
    h(:,m-1) = hist(samples,0:0.05:1)/N;
end
figure; hold on
handle = bar(0:0.05:1,h);
label = cell(1,4);
label{1} = '10^2 sets'; label{2} = '10^3 sets'; label{3} = '10^4 sets'; label{4} = '10^5 sets';
legend(handle,label)
title('distibution of the average of 4 samples')
%%
h = zeros(21,4);
for m=2:5
    N=10^m;
    samples = rand(N,5);
    samples = mean(samples,2);
    h(:,m-1) = hist(samples,0:0.05:1)/N;
end
figure; hold on
handle = bar(0:0.05:1,h);
label = cell(1,4);
label{1} = '10^2 sets'; label{2} = '10^3 sets'; label{3} = '10^4 sets'; label{4} = '10^5 sets';
legend(handle,label)
title('distibution of the average of 5 samples')
%%
% The Distribution becomes closer to normal distribution as the sample size
% increase. I can't judge how much it looks like a normal distribution by
% simply looking at it. 
%%
% * c)
%%
% A normal distribution should fit a unity line in normplot. figure
N = 1000;
samples = randn(N,1);
normplot(samples)
%%
% For small sample size, the normplot is far from a unity line at both
% tails of the distribution. As sample size increase, it's closer and
% closer. Also the slope gets steeper, reflecting a reduction of variance. 
% By a dozen of samples, it's hard to tell it apart from a normal
% distribution. 
%%
clear handle
N = 100000;
figure; hold on
n = 2;
samples = rand(N,n);
samples = mean(samples,2);
handle(:,1) = normplot(samples);
set(handle(1,1),'Marker','.')
n = 3;
samples = rand(N,n);
samples = mean(samples,2);
handle(:,2) = normplot(samples);
set(handle(1,2),'Marker','+')
n = 5;
samples = rand(N,n);
samples = mean(samples,2);
handle(:,3) = normplot(samples);
set(handle(1,3),'Marker','o')
n = 8;
samples = rand(N,n);
samples = mean(samples,2);
handle(:,4) = normplot(samples);
set(handle(1,4),'Marker','x')
n = 13;
samples = rand(N,n);
samples = mean(samples,2);
handle(:,5) = normplot(samples);
set(handle(1,5),'Marker','s')
n = 21;
samples = rand(N,n);
samples = mean(samples,2);
handle(:,6) = normplot(samples);
set(handle(1,6),'Marker','*')
xlim([0 1])
legend(handle(1,:),{'2','3','5','8','13','21 samples'},'Location','northwest')
%%
%% 4 Multi-dimensional Gaussians
%%
% * a)
%%
%<include>ndRandn.m</include>
%%
% * b)
%%
% If the 2-D Gaussian Distribution has mean m and covaiance S, then the
% marginal distribution has mean 
%%
% $$ \hat{u}^T m $$
%%
% and variance
% $$ \hat{u}^T S \hat{u} 
%%
% generate 10000 pairs of samples
N = 10000;
m = [1,3]';
covar = [9,4;4,16];
samples = ndRandn(m,covar,N);
m_s = mean(samples,2);
cov_s = (samples-m_s)*(samples-m_s)'/1000;
disp('theoretical mean:')
disp(m)
disp('sample mean:')
disp(m_s)
disp('theoretical covariance:')
disp(covar)
disp('sample covariance:')
disp(cov_s)
%%
% sample statistics is close to theoretical ones
%%
theta = (0:47)*2*pi/48;
us = [cos(theta);sin(theta)]; % generate a circle of unit vectors
samples_p = us'*samples; % project samples along unit vectors
mean_p = mean(samples_p,2); 
var_p = var(samples_p,0,2); % calculate mean and variance of projected samples
%%
figure; 
subplot(2,1,1);hold on
title('comparation of sample stats and prediction')
stem(theta,mean_p);
stem(theta,us'*m);
ylabel('mean')
legend('sample mean', 'predicted mean')
xticks((0:3)*2*pi/4)
xlim([0,2*pi])
subplot(2,1,2);hold on
stem(theta,var_p);
stem(theta,diag(us'*covar*us))
ylabel('variance')
xticks((0:3)*2*pi/4)
xlim([0,2*pi])
xlabel('projection direction / rad')
legend('sample variance', 'predicted variance','Location','southeast')
%%
%
%%
% * c)
%%
% Now we transform a unit circle to a ellipse in the same way we transform a standard
% indepedent multi-dimensional Gaussian samples to Gaussian with desired
% mean and covariance
%%
figure; hold on; grid on
[U,S,~] = svd(covar);
cov_sqrt = U*sqrt(S);
eus = 2*cov_sqrt*[us,us(:,1)]; % ellipse w/o adding mean
plot(eus(1,:)+m_s(1),eus(2,:)+m_s(2),'LineWidth',2)
scatter(samples(1,:),samples(2,:))
legend('ellipse','data')
%%
% This ellipse largely captures the shape of the data cloud. 
%%
% * d)
%%
% The largest variance is along the direction of the first eigenvector of
% covariance matrix
%%
u = U(:,1); % first eigenvector/sigular vector for symmetric matrix
angle = atan2(u(2),u(1)); 
angle = mod(angle,2*pi);
angle2 = mod(angle+pi,2*pi);
var_pu = var(u'*samples,0,2);
%%
figure; hold on
plot(theta,var_p);
plot(theta,diag(us'*covar*us))
stem([angle,angle2],[var_pu,var_pu],'r*')
ylabel('variance')
xticks((0:3)*2*pi/4)
xlim([0,2*pi]) 
xlabel('projection angle / rad')
legend('sample variance', 'predicted variance','predicted maxima','Location','southeast')
%%
figure; hold on; grid on
[U,S,~] = svd(covar);
cov_sqrt = U*sqrt(S);
eus = 2*cov_sqrt*[us,us(:,1)]; % ellipse w/o adding mean
plot(eus(1,:)+m(1),eus(2,:)+m(2),'LineWidth',2)
scatter(samples(1,:),samples(2,:))
% uu = cov_sqrt*u;
quiver(m(1),m(2),u(1),u(2),10,'Color','b')
legend('ellipse','data','direction where we can get largest variance')

