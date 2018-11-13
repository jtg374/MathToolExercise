close all; clear;
%% 1 Bayes rule and eye color
%%
% Let blue eye gene = b, brown eye gene = B. given Father has blue eyes.
%%
% 
% $$ P(Father = bb) = 1 $$
% 
%%
% given mother has brown eyes, 
%%
% 
% $$ P(Mother = BB) = P(Mother = Bb) = 1/2 $$
% 
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
mu = 5; % spike/interval
p = (mu.^k * exp(-mu))./factorial(k);
p = p/sum(p);
%%
% <include>randp.m</include>
%%
h = zeros(21,4);
for m=2:5
    N=10^m;
    samples = randp(p,N);
    h(:,m-1) = hist(samples,k)/N;
end
figure; hold on
plot(k,p)
bar(k,h)
xlabel('k')
ylabel('freq')
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
mu1 = 2;
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
plot(0:40,pq)
bar(0:40,h)
xlabel('k1+k2')
ylabel('freq')
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
bar(0:0.05:1,h)
%%
% I get a triangle shape, because that's essentially the sum of two
% independent random numbers divided by 2. there are more instances where
% two number add up to the center value. 
%%
% We can also think of that as the convolution of two uniform
% distributions. 
%%
% * b)
%%
h = zeros(21,4 );
for m=2:5
    N=10^m;
    samples = rand(N,3);
    samples = mean(samples,2);
    h(:,m-1) = hist(samples,0:0.05:1)/N;
end
figure; hold on
bar(0:0.05:1,h)
%%
% The distribution becomes
%%
h = zeros(21,4 );
for m=2:5
    N=10^m;
    samples = rand(N,4);
    samples = mean(samples,2);
    h(:,m-1) = hist(samples,0:0.05:1)/N;
end
figure; hold on
bar(0:0.05:1,h)
%%
h = zeros(21,4 );
for m=2:5
    N=10^m;
    samples = rand(N,5);
    samples = mean(samples,2);
    h(:,m-1) = hist(samples,0:0.05:1)/N;
end
figure; hold on
bar(0:0.05:1,h)

%%
% * c)
figure; hold on
N = 1000;
samples = randn(N,1);
normplot(samples)
n = 100;
samples = rand(N,n);
samples = mean(samples,2);
normplot(samples)
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
% generate 1000 samples
m = [1,3]';
S = [9,4;4,16];
samples = ndRandn(m,S,1000);
%%
theta = (0:47)*2*pi/48;
us = [cos(theta);sin(theta)];
samples_p = us'*samples;
mean_p = mean(samples_p,2);
var_p = var(samples_p,0,2);
figure; title('comparation of sample stats and prediction')
subplot(2,1,1);hold on
stem(theta,mean_p);
stem(theta,us'*m);
legend('sample mean', 'predicted mean')
xticks((0:3)*2*pi/4)
xlim([0,2*pi])
subplot(2,1,2);hold on
stem(theta,var_p);
stem(theta,diag(us'*S*us))
xticks((0:3)*2*pi/4)
xlim([0,2*pi])
xlabel('angle(u)/rad')
legend('sample variance', 'predicted variance','Location','southeast')
%%
%
%%
% * c)
%%
figure; hold on; grid on
scatter(samples(1,:),samples(2,:))
m_s = mean(samples,2);
cov_s = (samples-m_s)*(samples-m_s)'/1000;
eus(1,:) = us(1,:).*sqrt(var_p)'*2 + m_s(1); % us is the unit circle. 
eus(2,:) = us(2,:).*sqrt(var_p)'*2 + m_s(2);% rescale it with 2 std to get a ellipse. 
plot(eus(1,:),eus(2,:))


