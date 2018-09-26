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
matchingLight = P*matches
%% 
% And the actually testing wavelength spectrum is 
randomLight
%%
figure;hold on
plot(randomLight)
plot(matchLight)
legned('randomLight','matchLight')
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
absorptionRandomLight = cone*randomLight
absorptionMatchLight = cone*matchingLight
%%
% because for any pair of light ($l_1,l_2$) that map to the same knob settings
%%
% 
% $$M l_1 = M l_2$$
% 
%%
% In other words, $l_1-l_2$ is in M's null space. 
%%
% If 

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
p0 = ones(13,1);
p1 = [p0,x,y];
x2 = x.^2;
y2 = y.^2;
xy = x.*y;
p2 = [p1,x2,xy,y2];
% regression
beta0 = (p0'*p0)\p0'*z;
beta1 = (p1'*p1)\p1'*z;
beta2 = (p2'*p2)\p2'*z;
%%
% * order 0
z_hat = p0*beta0;
figure;hold on
plot3(x,y,z)
Z = reshape(z_hat,13,13);
surf(X,Y,Z)
rotate on
%%
% * order 1
z_hat = p1*beta1;
figure;hold on
plot3(x,y,z)
Z = reshape(z_hat,13,13);
surf(X,Y,Z)
rotate on
%%
% * order 2
z_hat = p0*beta0;
figure;hold on
plot3(x,y,z)
Z = reshape(z_hat,13,13);
surf(X,Y,Z)
rotate on






    
