close all; clear
%% 1 Trichromacy
load colMatch.mat;
%%
% * a)
%%
% Subjects match a random spectrum to the 3 primaries with 3 knob settings: 
N=31;
randomLight=rand(N,1);
matches=humanColorMatcher(randomLight,P)
%%
% What subject actually generated from primaries is
matchLight = P*matches
%% 
% And the actually testing wavelength spectrum is 
randomLight
%%
% Compare them in a single plot: 
figure;hold on
plot(randomLight)
plot(matchLight)
legend('randomLight','matchLight')
%%
% The two spectra look the same to the subject although they are acutually
% different,
%%
% because the dimension of the spectra (31) is much higher than human
% color perception (3). there are much more different spectra that have the
% same projection in lower dimension color perception space. 
%%
% * b)
%%
% The human matcher can be modeled by a matrix $M$.
expLight=eye(N);
M=humanColorMatcher(expLight,P)
%%
% * Verification
for ii = 1:5
    disp(['test light',num2str(ii)])
    randomLight=rand(N,1);
    matchesFunc=humanColorMatcher(randomLight,P)
    matchesMatx=M*randomLight
    disp('they are same!\n\n ')
end
%% 
% * c)
figure;
plot(Cones')
legend('L (red)','M (green)','S (blue)')
%%
% for an random wavelength spectrum
randomLight=rand(N,1)
%% 
% The subject match it with another spectrum generated from primaries
matches=M*randomLight;
matchingLight = P*matches
%%
% Two spectra produce equal cone absorption
absorptionRandomLight = Cones*randomLight
absorptionMatchLight = Cones*matchingLight
%%
% because for any pair of light ($l_1,l_2$) that map to the same knob settings
%%
% 
% $$M l_1 = M l_2$$
% 
%%
% In other words, $l_1-l_2$ is in $M$'s null space. 
%%
% If cones matching matrix $C$ has the same null space as $M$, then
%%
%
% $$C (l_1-l_2)$$
%
%%
% would also hold true, which means that any pair of light that elicits the
% same behavioral response, i.e, knob settings, produces the same cone
% absorption, vice versa. 
%%
% From SVD we can get the two null space, 
[~,~,V_M]=svd(M);
[~,~,V_cone]=svd(Cones);
null_M = V_M(:,4:end)
null_cone = V_M(:,4:end)
%%
% the two null space are the same because there will be no more addtional
% dimensions when we add columns from one to the other
svd([null_M,null_cone])
%%
% There are still 28 = 31-3 non-zero singular values, as well as two null
% space respectively. 
%%
% Alternatively, we can think of an arbitary spectrum $l$ and the spectrum
% that the subject match it with knob settings $x$, which generate a
% combination of primaries, $l'$.
%%
%
% $$l' = P x$$
%
%%
% Both spectra should produce the same cone absorption
%%
%
% $$C l = C P x$$
%
%%
% for every $l$. 
%%
% There should always be a unique matching. So 
%%
%
% $$x = (C P)^{-1} C l$$
%
%%
% So $CP$ should be invertible, and $\hat{M}=(C P)^{-1} C$ is actually the color
% matching matrix, $M$
M_hat = (Cones*P)\Cones;
error = M-M_hat;
all(all(error<1e-10)) % I don't use ==0 in order to avoid floatin point errors
%%
% which should have the same null space as C,
%%
% because for any
% $l_0$ in $C$'s null space,
%%
% $$C l_0 = 0$$
%%
% $$M l_0 = (C P)^{-1} \cdot C l_0 = 0$$
%%
% * d)
%%
% compare the responses between norm subjects and the patient.
randomLight=rand(N,5); % generate several test lights
matchesNorm =   humanColorMatcher(randomLight,P) % knob settings from normal subject
matchesAlt = altHumanColorMatcher(randomLight,P) % from the patient
%%
% They are totally different. I can't tell the pattern. 
%%
% Cone absorptions for test light are
Cones * randomLight
%%
% and cone absorptions for mixtures of matching primaries (normal) are
Cones * P * matchesNorm
%%
% Same with that of the test light. 
%%
% While cone absorptions for mixtures of matching primaries (patient) are
Cones * P * matchesAlt
%%
% For red and blue cones, absorptions are the same but green is different,
%So the patient may miss copies of green cone. 
%% 2 2D polynomial regression
load regress2.mat
%%
% * a)
x=D(:,1);y=D(:,2);z=D(:,3);
X=reshape(x,13,13);
Y=reshape(y,13,13);
Z=reshape(z,13,13);
figure;surf(X,Y,Z)
rotate3d on
%%
% * b)
% prepare predictors
p0 = ones(169,1);
p1 = [p0,x,y];
x2 = x.^2;
y2 = y.^2;
xy = x.*y;
p2 = [p1,x2,xy,y2];
x3 = x.^3;
y3 = y.^3;
x2y= x2.*y;
xy2= x.*y2;
p3 = [p2,x3,x2y,xy2,y3];
% regression
beta0 = (p0'*p0)\p0'*z;
beta1 = (p1'*p1)\p1'*z;
beta2 = (p2'*p2)\p2'*z;
beta3 = (p3'*p3)\p3'*z;
%%
% * order 0
z_hat = p0*beta0;
figure;hold on
scatter3(x,y,z)
Z = reshape(z_hat,13,13);
surf(X,Y,Z)
view(3)
rotate3d on
%%
% * order 1
z_hat = p1*beta1;
figure;hold on
scatter3(x,y,z)
Z = reshape(z_hat,13,13);
surf(X,Y,Z)
view(3)
rotate3d on
%%
% * order 2
z_hat = p2*beta2;
figure;hold on
scatter3(x,y,z)
Z = reshape(z_hat,13,13);
surf(X,Y,Z)
view(3)
rotate3d on
%%
% * order 3
z_hat = p3*beta3;
figure;hold on
scatter3(x,y,z)
Z = reshape(z_hat,13,13);
surf(X,Y,Z)
view(3)
rotate3d on
%% 
% 3rd order fit seems reasonable enough to capture the tilde-like trend of the data
%%
% If we plot the 
SE = (z-z_hat).^2;
figure;hist(SE)
%%
% mean squared error
MSE = mean(SE)
%%
z_hat_allTerm = p3*diag(beta3); % decompose z_hat, one term for every predictors
importance=sqrt(sum(z_hat_allTerm.^2,1)) % calculate vector length of each term
%%
% term $x^2y$, $x^2$ and $y^3$ terms are not so important
p3_drop = p3(:,importance>0.05);
beta3_drop = (p3_drop'*p3_drop)\p3_drop'*z;
z_hat_drop = p3_drop*beta3_drop;
MSE_new = mean((z-z_hat_drop).^2)
%%
% mean squared error only increase about 1%
%% 3 Constrained Least Squares
load constrainedLS.mat
%%
% * a)
%%
% The original problem can be written as
%%
% 
% $$\min_{\vec{v}} \vec{v}^T D^T D \vec{v}$$
% 
%%
% s.t. $\vec{v}^T\vec{w}=1$, where nth row in $D$ is $\vec{d}_n
%%
% singular value decompose $D=USV^T$, keep only first 2 colomns of U and
% first two rows of S
[U,S,V] = svd(data,'econ');
%%
% let $\tilde{v} = \tilde{S}V^T \vec{v}$, and \tilde{w} = \tilde{S}^{-1} V^T \vec{w}$
w_tilde = S\V'*w;
%%
% thus 
%%
% $$\min_{\tilde{v}} ||\tilde{v}||^2  $$
%%
% s.t. $\tilde{v}^T\tilde{w} = \vec{v}^T\vec{w}=1$.
%%
% * b)
%%
v_tilde = w_tilde/norm(w_tilde)^2
%%
figure; hold on
scatter(U(:,1),U(:,2),'k+') % first two columns of U is just transformed D
quiver(0,0,w_tilde(1),w_tilde(2),1,'r','LineWidth',2)
xx = v_tilde(1)+w_tilde(2)*(-250:250);
yy = v_tilde(2)-w_tilde(1)*(-250:250);
plot(xx,yy,'r--')
quiver(0,0,v_tilde(1),v_tilde(2),1,'b')
leg=legend('data','$\tilde{w}$','constraint line','$\tilde{v}$');
set(leg,'Interpreter','latex')
axis equal
xlim([-10,15])
ylim([-20,5])
hold off
%%
% * c)
%%
v = V/S*v_tilde
%%
figure; hold on
scatter(data(:,1),data(:,2),'k+') % first two columns of U is just transformed D
quiver(0,0,w(1),w(2),1,'r','LineWidth',2)
tt= w/norm(w)^2 + [w(2);-w(1)]*(-5:5);
plot(tt(1,:),tt(2,:),'r--')
quiver(0,0,v(1),v(2),1,'b','LineWidth',2)
leg=legend('data','$\vec{w}$','constraint line','$\vec{v}$');
set(leg,'Interpreter','latex')
axis equal
% xlim([-.25,.4])
% ylim([-.25,.4])
hold off
%%
% $\vec{v}$ is not perpendicular to constraint line in the original space,
% although is still on the constraint line. 
%%
% Total least square solution:
v_tls = V(:,end);
%%
figure; hold on
scatter(data(:,1),data(:,2),'k+') % first two columns of U is just transformed D
quiver(0,0,w(1),w(2),1,'r','LineWidth',2)
plot(tt(1,:),tt(2,:),'r--')
quiver(0,0,v(1),v(2),1,'b','LineWidth',1)
quiver(0,0,v_tls(1),v_tls(2),1,'c','LineWidth',2)
leg=legend('data','$\vec{w}$','constraint line','$\vec{v}$','$\vec{v}_{total\ least\ square}$');
set(leg,'Interpreter','latex')
axis equal
% xlim([-.25,.4])
% ylim([-.25,.4])
hold off
%%
% Solutions are different. 
%% 4 Principal Components
load PCA.mat
%%
% * a)
%%
figure;
plot(M)
xlabel('time')
ylabel('mean spike count')
%%
% There are several clusters of neuron, within which cell responses are
% similar. For example, there are 4 neurons that linearly ramp up slowly
% from beginning and peak at about 30th intervel and than ramp down. There
% are 4 neurons that elicit large peak at the middle of the trail. There
% are are a bunch of neurons that response weakly throughout the trail. 
%%
% * b)
%%
M_ = M - repmat(mean(M),50,1); % substract mean
[U,S,~] = svd(M_);S = diag(S);
figure;bar(S)
t=title('singular values of $\tilde{M}$');
set(t,'Interpreter','latex')
%%
% "True" dimensionality of the response should be 3. 
%%
% * c)
%%
figure;
plot(U(:,1:3))
xlabel('time')
leg = legend('first eigenvector of $\tilde{M}\tilde{M}^T$',...
    'second eigenvector of $\tilde{M}\tilde{M}^T$',...
    'third eigenvector of $\tilde{M}\tilde{M}^T$');
set(leg,'Interpreter','latex')
%%
% first 3 eigenvectors looks like half integer sinusoids. 
%%
% the fourth eigenvector
%%
figure;
plot(U(:,4))
xlabel('time')
legend('fourth eigenvector')
%%
% looks messy. 
%%
% * d)
%%
figure;
plot3(S(1)*U(:,1),S(2)*U(:,2),S(3)*U(:,3),'Marker','o');
xlabel('PC1');ylabel('PC2');zlabel('PC3')
view(3)
grid on
rotate3d on
%%
% The trajectory forms a loop, composed of two half circle connected with
% an angle in the PC3 dimension.
%%
close all





    
