clear all; close all
%% 1 Testing for (non)linearity
%%
% System 1
% is not linear.
% If it's linear, input 3 will generate 3 times of output of input 1,
% which is [3 12]
%%
% System 2
% is possibly linear.
L = [-3 0;1 0] 
%%
% is not unique because there are two elements in every output but there is only
% one independet input vector.
%%
% say
L = [0 -1.5;0 0.5]
%%
% verification
L*[2;4]
L*[-1;-2]
%%
% System 3
% is possibly linear.
L = [-1 1/3]
%%
% and is unique. 
%%
% verification
L*[2;6]
L*[-1;3]
%%
% System 4
% is not linear because all linear system return zero or zero vector in
% response to zero input.
%%
% System 5
% is not linear because it doesn't follow the rule of superposition
[1 -1] + 2*[1 1] 
%%
% where output
[3  2] + 2*[1 2]
%%
% is not equal to [5 3]

%% 2 Inner product with a unit vector
%%
% * a) 
%%
% $$ \hat{u} (\vec{v} \cdot \hat{u}) $$
funcA = @(u,v) u*dot(v,u);
%%
% * b) 
%%
% $$ \vec{v} - \hat{u} (\vec{v} \cdot \hat{u}) $$
funcB = @(u,v) v - funcA(u,v);
%%
% * c)
%%
% $$ ||\vec{v} - \hat{u} (\vec{v} \cdot \hat{u})||^2 $$
funcC = @(u,v) sqrt(sum(func(u,v).^2));

%% 
% * 2.1 2-dimensional test

v = rand(2,1)*2-1
a = rand()*2*pi; u = [cos(a);sin(a)]

figure; hold on
quiver(0,0,v(1),v(2),1,'b-','DisplayName','$\vec{v}$')
quiver(0,0,u(1),u(2),1,'r-','DisplayName','$\hat{u}$')
p=funcA(u,v)
quiver(0,0,p(1),p(2),1,'b-.','DisplayName','$\hat{u} (\vec{v} \cdot \hat{u})$')
q=funcB(u,v)
quiver(p(1),p(2),q(1),q(2),1,'b:','DisplayName','$\vec{v} - \hat{u} (\vec{v} \cdot \hat{u})$')
leg=legend();
set(leg,'Interpreter','latex')
plot([-2,2],[0,0],'k','HandleVisibility','off')
plot([0,0],[-2,2],'k','HandleVisibility','off')
hold off
axis equal
xlim([-1.5,1.5])
ylim([-1.5,1.5])
%%
% The codes are working. 
%% 
% * 2.2 4-dimensional test
v = rand(4,1)*2-1
u0 = rand(4,1)*2-1; u = u0/norm(u0)
%%
% the vector in a)
p = funcA(u,v)
%%
% is in the same direction as $\hat{u}$.
p./u
%%
% the vector in b)
q = funcB(u,v)
%%
% is orthogonal to $\vec{p}$.
dot(p,q)
%%
% almost zero.
%%
% the sum of the two vector
p+q
%%
% is equal to $\vec{v}$.
v
%%
% the sum of squared length of the two vectors
sum(p.^2) + sum(q.^2)
%%
% is equal to $ \vec{v} ^2 $. 
%%
sum(v.^2)

%% 3 Geometry of linear transformation
%%
% <include>vecLenAngle.m</include>
%%
% singular value decomposition of a random matrix M
M = rand(2,2)
[U,S,V] = svd(M)
%%
% a unit circle to be operated by the matrix
theta = (0:64)/64*2*pi;
P = [cos(theta);sin(theta)];
figure;hold on
plot(P(1,1),P(2,1),'r','Marker','*')
plot(P(1,:),P(2,:))
hold off;axis equal
xlim([-1.5,1.5])
ylim([-1.5,1.5])
%%
% *First transformation*
R1 = V'*eye(2)
[lu,lv,a] = vecLenAngle(R1(:,1),R1(:,2));
disp(['resulting length ',num2str(lu),', ',num2str(lv), '. Angle ',num2str(a*180/pi),' degree'])
%%
% Both angle and lengths are preserved. 
plotVec2(R1);
%%
P = V'*P;
figure;hold on
plot(P(1,1),P(2,1),'r','Marker','*')
plot(P(1,:),P(2,:))
hold off;axis equal
xlim([-1.5,1.5])
ylim([-1.5,1.5])
%%
% the circle rotated (potentially flipped axis addtionally) because V is an
% orthogonal matrix.
%%
% *Second transformation*
R2 = S*R1
[lu,lv,a] = vecLenAngle(R2(:,1),R2(:,2));
disp(['resulting length ',num2str(lu),', ',num2str(lv), '. Angle ',num2str(a*180/pi),' degree'])
%%
% Both angle and length are changed 
plotVec2(R2);
%%
P = S*P;
figure;hold on
plot(P(1,1),P(2,1),'r','Marker','*')
plot(P(1,:),P(2,:))
hold off;axis equal
xlim([-1.5,1.5])
ylim([-1.5,1.5])
%%
% The circle was stretched into an oval because S is a diagonal matrix. 
%%
% *Third transformation*
R3 = U*R2
[lu,lv,a] = vecLenAngle(R3(:,1),R3(:,2));
disp(['resulting length ',num2str(lu),', ',num2str(lv), '. Angle ',num2str(a*180/pi),' degree'])
%%
% Both angle and lengths are preserved.
plotVec2(R3);
%%
P = U*P;
figure;hold on
plot(P(1,1),P(2,1),'r','Marker','*')
plot(P(1,:),P(2,:))
hold off;axis equal
xlim([-1.5,1.5])
ylim([-1.5,1.5])
%%
% the oval rotated (potentially flipped axis addtionally) because U is an
% orthogonal matrix.

%% 4 A simple visual neuron
%%
% * a) 
% the system is not linear because it only takes positive input,
% but with positive inputs the response is essentially the dot product
% a vector of 7 intensities values of each location with the weight vector
w = [1,3,4,5,4,3,1]
%%
% * b)
% the unit vector that can generate largest response is parallel with the
% weight vector
u = w/norm(w)
%%
% because the response dot(u,w), which is equal to $|w|cos(\theta)$, where a is
% the angle between u and w, takes largest value only when $cos(\theta) = 1$, which 
% means u and w are on the same direction. 
%%
% * c)
%%
% Answer: [1,0,0,0,0,0,0] or [0,0,0,0,0,0,1]. 
%%
% Proof:
%%
% 1. In two dimensional case
%%
% Let input vector $u = (cos(\alpha),sin(\alpha))$ where $0<=\alpha<=\pi/2$
% and weight vecor $w = (w_1,w_2)$.
%%
% Then response equals $cos(\alpha)*w_1 + sin(\alpha)*w_2 = |w|cos(\theta)$, where
% $\theta = \alpha - tan^{-1}(w_2/w_1)$.
%%
% The response is the least when $\theta$ is most further away from 0, that
% is w1 if w1<=w2 (take $\alpha = 0$), or w2 if w2<w1 (take $\alpha = \pi/2$).
% so $cos(\alpha)*w_1 + sin(\alpha)*w_2 >= min(w_1,w_2)$ and is the minimum 
% (takes minimum when $\alpha = 0$ or $\pi/2$). 
%%
% 2. three dimensional case
%%
% Let input vector $u = 
% (cos(\alpha_1), sin(\alpha_1)*cos(\alpha_2) ), sin(\alpha_1)*sin(\alpha_2) )$
% and $w = (w_1,w_2,w_3)$.
% Then the response =
%%
% $cos(\alpha_1)*w_1 + sin(\alpha_1)*( cos(\alpha_2)*w_2 + sin(\alpha_2)*w_3 )$
%%
% whose miminum (if exists) should be equal to 
% $cos(\alpha_1)*w_1 + sin(\alpha_1)*min(w_2,w_3)$
%%
% (when $\alpha_2 = 0$ or $\pi/2$)
%%
% whose miminum exists and is equal to 
% $min(w_1,min(w_2,w_3))$
% $= min(w_1,w_2,w_3)$
%%
% (when $\alpha_1 = 0$ or $\pi/2$)
%%
% so when only one of the input element (corresponding to the minimum in weight vector)
% is 1 and others equal 0, the response takes minimum, which is $=
% min(w_1,w_2,w_3)$. 
%%
% 3.. n dimensional case (n>2) (induction)
%%
% Let input vector $u = (cos(\alpha_1),sin(\alpha_1)*cos(\alpha_2),sin(\alpha_1)*
% sin(\alpha_2)*cos(\alpha_3),...,sin(\alpha_1)*sin(\alpha_2)*...*sin(\alpha_{n-1}))$
% and $w = (w_1,w_2,w_3,...w_n)$. 
%%
% Then response equals 
% $cos(\alpha_1)*w_1 + sin(\alpha_1)*(cos(\alpha_2)*w_2+sin(\alpha_2)*(...
% +(cos(\alpha_{n-1})*w_{n-1} + sin(\alpha_{n-1})*w_n))))$
% $\geq cos(\alpha_1)*w_1 + sin(\alpha_1)*(cos(\alpha_2)*w_2+sin(\alpha_2)*
% min(w_{n-1},w_n)$
% $\geq ... \geq min(w_1,w_2,w_3,...,w_n)$
%%
% (equal if every $alpha_i$ is either 0 or $\pi/2$)
%%
% The right side is the minimum and exists. 
%%
% Q.E.D.  
%% 5 Gram-Schmidt
%%
% <include>gramSchmidt.m</include>
%%
% * 3d plot
Q = gramSchmidt(3)
figure; hold on
quiver3(0,0,0,Q(1,1),Q(2,1),Q(3,1))
quiver3(0,0,0,Q(1,2),Q(2,2),Q(3,2))
quiver3(0,0,0,Q(1,3),Q(2,3),Q(3,3))
axis equal
xlim([-1.1 1.1]);ylim([-1.1 1.1]);zlim([-1.1 1.1]);
grid on
view(3)
% rotate3d on
%%
% * 7d test
Q = gramSchmidt(7)
Q * Q'
%%
% $QQ^T$ is almost idendity matrix, so Q has orthonormal columes. 
%% 6 Null and Range spaces
%%
% <include>mtxNull.m</include>
%%
% <include>mtxRange.m</include>
%%
% <include>mtxInverse.m</include>
%%
load mtxExamples.mat
%%
% * MTX1
% Null Space
nullVec = mtxNull(mtx1)
%%
if nullVec
    isZero = mtx1*nullVec
end
%%
% Range Space
y = mtxRange(mtx1)
%%
if y
    x = mtxInverse(mtx1)*y
end
%%
y_hat = mtx1*x
%% 
% is equal to y. 
%%
% * mtx2
% Null Space
nullVec = mtxNull(mtx2)
%%
if nullVec
    isZero = mtx2*nullVec
end
%%
% Range Space
y = mtxRange(mtx2)
%%
if y
    x = mtxInverse(mtx2)*y
end
%%
y_hat = mtx2*x
%% 
% is equal to y. 
%%
% * mtx3
% Null Space
nullVec = mtxNull(mtx3)
%%
if nullVec
    isZero = mtx3*nullVec
end
%%
% Range Space
y = mtxRange(mtx3)
%%
if y
    x = mtxInverse(mtx3)*y
end
%%
y_hat = mtx3*x
%% 
% is equal to y. 
%%
% * mtx4
% Null Space
nullVec = mtxNull(mtx4)
%%
if nullVec
    isZero = mtx4*nullVec
end
%%
% Range Space
y = mtxRange(mtx4)
%%
if y
    x = mtxInverse(mtx4)*y
end
%%
y_hat = mtx4*x
%% 
% is equal to y. 
%%
% * mtx5
% Null Space
nullVec = mtxNull(mtx5)
%%
if nullVec
    isZero = mtx5*nullVec
end
%%
% Range Space
y = mtxRange(mtx5)
%%
if y
    x = mtxInverse(mtx5)*y
end
%%
y_hat = mtx5*x
%% 
% is equal to y. 
