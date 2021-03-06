close all; clear
%% Dueling estimators
%%
% In this problem, we use simulation to compare three estimators of the mean of a Normal (Gaussian) distribution.
%%
NN = 1e4;
N = 10;
S = randn(N,NN); % generate NN samples each of size N
%%
% * a)
%%
% First consider the average, which minimizes the sum of squared deviations, and is also the Maximum Likelihood estimator.
%%
avg = mean(S); % average of each of the NN samples
figure; hist(avg,50)
xlabel('mean(x)');ylabel('counts')
xlim([-2.3,2.3])
%%
% The histogram has a bell shape because of the central limit theorem, with
% a variance: 
%%
% $$ \sigma_s^2 = 1^2/N = 0.1 $$
%%
disp('variance of mean: ')
disp(var(avg)) % var() use N-1, unbiased estimator
%%
% This is close to 0.1. 
%%
% * b)
%%
% Now consider the median, which minimizes the sum of absolute deviations
%%
med = median(S); % median of each of the NN samples
figure; hist(med,50)
xlabel('median(x)');ylabel('counts')
xlim([-2.3,2.3])
%%
% The distribution has a bell shape slightly wider than the distribution of
% the mean. Compare it to a normal distribution: 
%%
figure; normplot(med)
%%
% The distribution of median doesn't deviate significantly from a Normal
% distribution. Compare the Q-Q plot for the median estimator to that for the mean 
%%
figure;qqplot(med,avg)
xlabel('median(x) quantile')
ylabel('mean(x) quantile')
%%
% The two distrubutions only deviate at the tails. Which is consistent with
% a change of dispersion. 
%%
% * c)
%%
% Finally, consider an estimator that computes the average of the minimum
% and maximum over the sample (as shown in class, this one minimizes the
% $L_\inf$?norm). 
%%
mid = (max(S)+min(S))/2; % range middle of each of the NN samples
figure; hist(mid,50)
xlabel('middle(x)');ylabel('counts')
xlim([-2.3,2.3])
%%
% This one also has a bell shape. Compare it to a normal distribution: 
%%
figure; normplot(mid)
%%
% This time we can see a systematic deviation outside [-1,1]. Compare it
% two previous distributions
%%
figure;qqplot(mid,avg)
xlabel('middle(x) quantile')
ylabel('mean(x) quantile')
%%
figure;qqplot(mid,med)
ylabel('median(x) quantile')
xlabel('middle(x) quantile')
%%
% Again, range middle distribution is wider than mean distribution, and is
% closer to median distribution. 
%%
% * d)
%%
% All three of these estimators are unbiased (because of the symmetry of
% the distribution), so we can use variance as the sole criterion for
% quality. 
%%
NN = 1e4;
N = 2^8;
S = randn(N,NN); % generate NN samples each of size N
%%
n = 2.^(3:8);
Vavg = zeros(size(n));
Vmed = zeros(size(n));
Vmid = zeros(size(n));
%%
for ii = 1:length(n)
    subS = S(1:n(ii),:);
    Vavg(ii) = var(mean(subS));
    Vmed(ii) = var(median(subS));
    Vmid(ii) = var((min(subS)+max(subS))/2);
end
%%
figure
loglog(n,Vavg)
hold on
loglog(n,Vmed)
loglog(n,Vmid)
loglog(n,1./n,'--');
legend('mean','median','range middle','1/N','Location','southwest')
xticks(n)
xlabel('subsample size (N)')
ylabel('estimator variance')
%%
% Only the variance of mean follows 1/N. variance of median drops with N at
% same rate but variance itself is always larger. variance of range middle
% decreases slower as N increases and is even larger
%%
% For N=256, range middle estimator has variance: 
%%
disp(Vmid(end))
%%
% This corresponds to a mean estimator of sample size N = 
disp(round(1/Vmid(end))) % according to theoretical variance
%%
% And a median estimator of approx. sample size N = 
%%
d = Vmed(2)/Vavg(2); % approx. difference of med and avg estimator around N=16
disp(round(d*1/Vmid(end)))
%%
close all
%% 2 Bayesian inference of binominal Proportions
%%
% Poldrack found that Broca�s area was reported activated in 103 out of 869
% fMRI contrasts involving engagement of language, but this area was also
% active in 199 out of 2353 contrasts not involving language. 
%%
% * a)
%%
% Assume that the conditional probability of activation given language, as
% well as that of activation given no language, each follow a Bernoulli
% distribution. 
%%
% $$ P(activation | language)= x_l $$
%%
% $$ P(activation | no language)= x_{nl}) $$
%%
% for all possible $x_l$, $x_{nl}$
x = (0:.001:1)';
%%
% The likelihood of observed frequencies is
%%
% $$ P( data | x_l ) = C^{869}_{103}x_l^{103}(1-x_l)^{766} $$
%%
lkh_l = nchoosek(869,103)*x.^103.*(1-x).^(869-103);
%%
% $$ P( data | x_h ) = C^{2353}_{199}x_{nl}^{199}(1-x_{nl})^{2154} $$
%%
lkh_nl = nchoosek(2353,199)*x.^199.*(1-x).^(2353-199);
%%
figure;
subplot(2,1,1);
bar(x,lkh_l);
ylabel('P(data | x_l)')
xlabel('x_l')
subplot(2,1,2);
bar(x,lkh_nl);
ylabel('P(data | x_{nl})')
xlabel('x_{nl}')
%%
% * b)
%%
% Find the value of x that maximizes each discretized likelihood function, given Poldrack�s observed frequencies of activation.  
[~,ind] = max(lkh_l);
x_l_max = x(ind);
[~,ind] = max(lkh_nl);
x_nl_max = x(ind);
disp('most likely activation probability in tasks involving language: ')
disp(x_l_max)
disp('most likely activation probability in tasks not involving language: ')
disp(x_nl_max)
%%
% For a Binominal probability, the Likihood function
%%
% $$ L \propto x^k (1-x)^{n-k} $$
%%
% $$ \frac{dL}{dx} \propto x^{k-1}(1-x)^{n-k-1}((n-k)x-k(1-x) $$
%%
% $$ = (x^{k-1}(1-x)^{n-k-1})(k-nx) $$
%%
% So $\frac{dL}{dx} > 0$ when $x < \frac{k}{n}$, and $\frac{dL}{dx} < 0$ when $x > \frac{k}{n}$
%%
% $x = \frac{k}{n}$ maximize L. 
%%
disp('most likely activation probability in tasks involving language: 103/869 = 0.1185')
disp('most likely activation probability in tasks not involving language: 199/2353 = 0.846')
%%
% The exact ML estimates matches the numerical ones. 
%%
%%
% * c)
%%
% compute and plot the discrete posterior distributions assume a uniform
% prior P(x). 
%%
% $$ P (x | data) = \frac{P(x)P(data|x)}{P(data)} %%
%%
% Since both P(x) and P(data) is irrelevant of x, posterior is just
% normalized likelihood
posterior_l = lkh_l/sum(lkh_l);
posterior_nl = lkh_nl/sum(lkh_nl);
%%
% cumulative distributions $P(X leq x | data)$
cdf_l = cumsum(posterior_l);
cdf_nl = cumsum(posterior_nl);
%%
% Use the cumulative distributions to compute (discrete approximations to)
% upper and lower 95% confidence bounds on each proportion. 
cl_l = x(cdf_l>0.05 & cdf_l<0.95);
cl_nl = x(cdf_nl>0.05 & cdf_nl<0.95);
disp('upper 95% confidence bound of x_l: ')
disp(cl_l(end))
disp('lower 95% confidence bound of x_l: ')
disp(cl_l(1))
disp('upper 95% confidence bound of x_nl: ')
disp(cl_nl(end))
disp('lower 95% confidence bound of x_nl: ')
disp(cl_nl(1))
%%
% Looks the lower 95% condifence of $x_l$ is higher than the upper 95%
% confidence bound of $x_{nl}$.
%%
% * d)
%%
% Given that these two frequencies are independent, the (discrete) joint
% distribution is given by the outer product of the two marginals. 
posterior = posterior_nl * posterior_l';
figure; hold on
imagesc(posterior)
plot([0 1]*1e3+1,[0 1]*1e3+1,'w--')
xlabel('x_l')
ylabel('x_{nl}')
set(gca,'XTick',(0:0.1:1)*1e3+1,'XTickLabel',(0:0.1:1))
set(gca,'YTick',(0:0.1:1)*1e3+1,'YTickLabel',(0:0.1:1))
axis equal
xlim([0 1]*1e3+1)
ylim([0 1]*1e3+1)
%%
% $$ P(x_l \leq x_{nl}) = \sum_x P(x_l \leq x | x_{nl} = x) P(x_{nl} = x) $$
%%
% $$ P(x_l > x_{nl}) = 1 - P(x_l \leq x_{nl}) $$
%%
larger_is_l = 1 - posterior_nl' * cdf_l; 
%%
% $$ P(x_nl \leq x_{nl}) = \sum_x P(x_nl \leq x | x_{l} = x) P(x_{l} = x) $$
%%
% $$ P(x_nl > x_{nl}) = 1 - P(x_nl \leq x_{nl})
%%
larger_is_nl = 1 - posterior_l' * cdf_nl; 
%%
% Probability that $x_l > x_{nl} $
%%
disp(larger_is_l)
%%
% Probability that $x_l \leq x_{nl} $
%%
disp(1-larger_is_l)
%%
% * e)
%%
% Compute the probability P(language | activation). To do this use the
% likelihoods from part (a) and assume that there is a uniform prior on whether language is engaged in a task or not i.e. P(language)= 0.5. 
%%
% Assume we know $x_l$ and $x_{nl}$, then
%%
% $$ P(language | activation) = \frac{P(activation | language )P( language )}{P(activation)} $$
%%
% where
%%
% $$ P(activation) = P(activation | language) P(language) + P(activation |
% no language) P (no language) = \frac{1}{2}(x_l + x_{nl})$$
%%
% $$ P(language | activation) = \frac{x_l}{x_l+x_{nl}} $$
%%
% So the expectation of this probability is 
%%
% $$\sum_{x_l} \sum_{x_{nl}} \frac{x_l}{x_l+x_{nl}} P(x_l) P(x_{nl}) $$
%%
[xl,xnl]=meshgrid(x,x);
CP = xl./(xl+xnl); % conditional probability of P(language | activation) given xl xnl
CP(isnan(CP)) = 0;
EP = posterior_nl'*CP*posterior_l; % expected probability of P(language | activation)
%%
% the expectation probability that observing activation in Broca�s area
% implies engagement of language processes is: 
%%
disp(EP)
%%
% is not much larger than 0.5, the prior probability that a contrast engage
% language. So Poldrack is right. 
%%
close all
%%
%% 3 Bayesian estimation
%%
% Tina and Perri are looking for Nikhil in a very large one-dimensional
% shopping mall. Location is specified by a coordinate X.
dx=1;
x = 0:dx:100;
%%
% Nikhil prefers to be near the center of the
% shopping mall at location 50. He has a prior Gaussian distribution
% centered on 50 with variance 40.
priorpdf = exp(-(x-50).^2/(2*40))/(sqrt(2*pi)*40);
priorcdf = normcdf(x,50,sqrt(40));
prior    = diff([priorcdf 1]);
%%
% Prior knowledge of Nikhil's location is
%%
% $$ f(X) = \mathcal{N}(50,40) $$
%% 
% The only clue they have is a coffee cup
% of a brand that only Nikhil drinks that they find at location X=30.
% The coffee cup is cold and Nikhil has wandered off. 
% Based on the location of the coffee cup,
% the likelihood function of his location is a Gaussian distribution
% with mean X=30 and variance 100.
lkhpdf = exp(-(x-30).^2/(2*100))/(sqrt(2*pi)*100);
lkhcdf = normcdf(x,30,sqrt(100));
likelihood = diff([lkhcdf 1]);
%%
% $$ f(coffee\ cup|X) = \mathcal{N}(30,100) $$
%%
% Use Bayes rule, the posterior distribution of Nikhil: 
%%
% $$ f(X | coffee\ cup) \propto f(X) f(coffee\ cup | X)$$
%%
% $$ \propto e^{-\frac{(x-50)^2}{80} - \frac{(x-30)^2}{200}}$$
%%
% $$ \propto e^{-\frac{(x-310/7)^2}{400/7}} $$
%%
% so
%%
% $$ f(X | coffee\ cup) = \mathcal{N}(310/7,200/7) $$
%%
normalization = sum(priorpdf.*lkhpdf)*dx;
posterior = priorpdf.*lkhpdf/ normalization;
%%
% posterior has a variance of 200/7 = 28.57
%%
figure; hold on
plot(priorpdf)
plot(lkhpdf)
plot(posterior)
legend('prior','likelihood','posterior')
xlabel('Nikhil''s location');
ylabel('probability density')
%%
% * b)
%%
% The coffee cup was not that cold after all. Nikhil�s likelihood function
% has mean X=30, but with a smaller variance of 20. 
lkhpdf = exp(-(x-30).^2/(2*20))/(sqrt(2*pi)*20);
lkhcdf = normcdf(x,30,sqrt(100));
likelihood = diff([lkhcdf 1]);
%%
normalization = sum(priorpdf.*lkhpdf)*dx;
posterior = priorpdf.*lkhpdf/ normalization;
%%
% The posterior moved leftwards toward the mean of likelihood. This makes
% sense because we are now more confident about the evidence, so our estimate is more biased to the coffee cup's position.  
%%
figure; hold on
plot(priorpdf)
plot(lkhpdf)
plot(posterior)
legend('prior','likelihood','posterior')
xlabel('Nikhil''s location');
ylabel('probability density')
%%
% * c)
%%
% If the prior had been uniform, the posterior would be proportional to the
% likelihood, thus having the same variance as the likelihood. 
%%
% The inclusion of prior lower the variance of posterior from 100 to 28.57,
% or add to the inverse of the variance. 
%%
% $$ \frac{1}{\sigma^2_{posterior}} = \frac{1}{\sigma^2_{prior}} + \frac{1}{\sigma^2_{likelihood}} $$
%% 4 Signal Detection Theory
%%
% * a)
%%
% For the �no coherence�� stimulus, generate 1000 trials of the firing rate
% of the neuron in response to these stimuli. Since we cannot have negative
% firing rates, set all rates that are below zero to zero. 
%%
% $$ f(r|No Coherence) = \mathcal{N}(5,1) $$
%%
N = 1000;
NC = 5+randn(N,1);
NC(NC<0)=0;
%%
% $$ f(r|right) = \mathcal{N}(8,1) $$
SC = 8+randn(N,1);
SC(SC<0)=0;
%%
figure;hold on
h1=histogram(NC);
h2=histogram(SC);
legend('0% coherence','10% coherence right')
xlabel('firing rate')
title('FR histogram')
%%
% * b)
%%
% $$ d' = (8-5)/1 = 3 $$
%%
d = (mean(SC) - mean(NC)) / ((std(SC)+std(NC))/2) 
%%
% * c)
%%
% Select various thresholds t and for each threshold calculate the hit and
% false-alarm rates using your sample data from (a). 
%%
t = 0:.1:12;
hit = 1 - cumsum(hist(SC,t))/N;
FA = 1 - cumsum(hist(NC,t))/N;
figure;plot(FA,hit,'k')
xlabel('False Alarm')
ylabel('Hit')
title('ROC')
axis equal
ylim([0 1]);xlim([0 1])
%%
% percentage-correct equals
%%
% $$ P(right) * P(r>threshold | right) + P(No Coherence) * P(r<threshold | No Coherence) $$
%%
% $$ = P(right) * Hit + P(No Coherence) * CR $$
%%
% assume that 0% and 10% coherence stimuli occur equally often. To maximize
% percentage-correct, we want to set a threshould that equals: 
[~,ind] = max(0.5*hit + 0.5*(1-FA));
threshold = t(ind)
%%
figure
subplot(2,1,1);plot(FA,hit,'k')
xlabel('False Alarm')
ylabel('Hit')
title('threshold assuming equal occurence')
hold on
axis equal
ylim([0 1]);xlim([0 1])
scatter(FA(ind),hit(ind),'k*')
plot(FA(ind)+[-1 1]*.5,hit(ind)+[-1 1]*.5,'k--')
subplot(2,1,2);
h1=histogram(NC);hold on
h2=histogram(SC);
xlabel('firing rate')
yl = ylim;
plot([threshold threshold],[0 1000],'k--','LineWidth',3)
ylim(yl)
legend('0% coherence','10% coherence right','threshold')

%%
% assume that 10% coherence stimuli occurs 75% of the time. To maximize
% percentage-correct, we want to set a threshould that equals: 
[~,ind] = max(0.75*hit + 0.25*(1-FA));
threshold = t(ind)
%%
figure
subplot(2,1,1);plot(FA,hit,'k')
xlabel('False Alarm')
ylabel('Hit')
title('threshold assuming 75% occurence')
hold on
axis equal
ylim([0 1]);xlim([0 1])
scatter(FA(ind),hit(ind),'k*')
plot(FA(ind)+[-1 1]*.75,hit(ind)+[-1 1]*.25,'k--')
subplot(2,1,2);
h1=histogram(NC);hold on
h2=histogram(SC);
xlabel('firing rate')
yl = ylim;
plot([threshold threshold],[0 1000],'k--','LineWidth',3)
ylim(yl)
legend('0% coherence','10% coherence right','threshold')

%%
% * d)
%%
% Consider now a neuron with a more �noisy� response so that the mean
% firing rates are the same but the standard deviation is 2 spikes/s instead of 1 spike/s. 
NC = 5+randn(N,1)*2;
NC(NC<0)=0;
SC = 8+randn(N,1)*2;
SC(SC<0)=0;
%%
% now d' is halved
d = (mean(SC) - mean(NC)) / ((std(SC)+std(NC))/2) 
%%
figure;hold on
h1=histogram(NC);
h2=histogram(SC);
legend('0% coherence','10% coherence right')
xlabel('firing rate')
title('FR histogram')

%%
% the ROC curve is shifted towards diagonal
t = 0:.1:12;
hit = 1 - cumsum(hist(SC,t))/N;
FA = 1 - cumsum(hist(NC,t))/N;
figure;plot(FA,hit,'k')
xlabel('False Alarm')
ylabel('Hit')
title('ROC')
axis equal
ylim([0 1]);xlim([0 1])
%%
% assume that 0% and 10% coherence stimuli occur equally often. To maximize
% percentage-correct, we want to set a threshould that equals: 
[~,ind] = max(0.5*hit + 0.5*(1-FA));
threshold = t(ind)
%%
% Still about at the center of two means
%%
figure
subplot(2,1,1);plot(FA,hit,'k')
xlabel('False Alarm')
ylabel('Hit')
title('threshold assuming equal occurence')
hold on
axis equal
ylim([0 1]);xlim([0 1])
scatter(FA(ind),hit(ind),'k*')
plot(FA(ind)+[-1 1]*.5,hit(ind)+[-1 1]*.5,'k--')
subplot(2,1,2);
h1=histogram(NC);hold on
h2=histogram(SC);
xlabel('firing rate')
yl = ylim;
plot([threshold threshold],[0 1000],'k--','LineWidth',3)
ylim(yl)
legend('0% coherence','10% coherence right','threshold')

%%
% assume that 10% coherence stimuli occurs 75% of the time. To maximize
% percentage-correct, we want to set a threshould that equals: 
[~,ind] = max(0.75*hit + 0.25*(1-FA));
threshold = t(ind)
%%
% It's now further leftwards. 
%%
figure
subplot(2,1,1);plot(FA,hit,'k')
xlabel('False Alarm')
ylabel('Hit')
title('threshold assuming 75% occurence')
hold on
axis equal
ylim([0 1]);xlim([0 1])
scatter(FA(ind),hit(ind),'k*')
plot(FA(ind)+[-1 1]*.75,hit(ind)+[-1 1]*.25,'k--')
subplot(2,1,2);
h1=histogram(NC);hold on
h2=histogram(SC);
xlabel('firing rate')
yl = ylim;
plot([threshold threshold],[0 1000],'k--','LineWidth',3)
ylim(yl)
legend('0% coherence','10% coherence right','threshold')

%%
close all
