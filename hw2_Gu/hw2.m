%% 1 Trichromacy
load colMatch.mat;
%%
% * a)
%%
% Subjects match a random spectrum to the 3 primaries
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
figure;hold on
plot(randomLight)
plot(matchLight)
legend('randomLight','matchLight')
%%
% The two spectra look the same to the subject although they are acutually
% different because the dimension of the spectra (31) is much higher than human
% color perception (3). there are much more different spectra that have the
% same projection in lower dimension space. 
%%
% * b)
expLight=eye(N);
M=humanColorMatcher(expLight,P)
%%
% * Verification
for ii = 1:5
    disp(['test light',num2str(ii)])
    randomLight=rand(N,1);
    matchesFunc=humanColorMatcher(randomLight,P)
    matchesMatx=M*randomLight
    disp('they are same! ')
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
% In other words, $l_1-l_2$ is in M's null space. 
%%
% If cones matching matrix C has the same null space as M, then
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
null_M = V_M(:,4:end);
null_cone = V_M(:,4:end);
%%
% the two null space are the same because there will be no more addtional
% dimensions when we add columns from one to the other
svd([null_M,null_cone])
%%
% There are still 28 = 31-3 non-zero singular values, as well as two null
% space respectively. 
%%
% Alternatively, we can think of an arbitary spectrum $l$ and the spectrum
% that the subject match it with primaries $l'$. 
%%
%
% $$l' = P x$$
%
%%
% where x is the knob settings. both spectra should produce the same cone absorption
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
% So $CP$ should be invertible, and $M=(C P)^{-1} C$ is actually the color
% matching matrix, 
M_hat = (Cones*P)\Cones;
error = M-M_hat;
all(all(error<1e-10))
%%
% which should have the same null space as C, because any
% $l_0$ in $C$'s null space, $C l_0 = 0$, $M l_0 = (C P)^{-1} \cdot 0 = 0$
%%
% * d)
randomLight=rand(N,5); % generate several test lights
matchesNorm =   humanColorMatcher(randomLight,P) % knob settings from normal subject
matchesAlt = altHumanColorMatcher(randomLight,P) % from the patient
%%
% They are totally different. I can't tell the pattern. 
%%
% Cone absorption for test Light
Cones * randomLight
%%
% Cone absorption for mixtures of matching primaries (normal)
Cones * P * matchesNorm
%%
% Same Cone absorption
%%
% Cone absorption for mixtures of matching primaries (patient)
Cones * P * matchesAlt
%%
% Cone absorption for red and blue cones are the same but green is random,
%So the patient may miss copies of green cone. 
%% 2D polynomial regression
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






    
