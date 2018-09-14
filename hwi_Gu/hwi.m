%% 2. Vector math.
%%
% * (a) 
clear all
close all
%%
% * (b) 
x = [3, 2, 5, 1]
%%
% * (c) 
y = zeros(1,4);
y(1)=-2; y(3)=2; y(4)=-4;
%%
% * (d) 
2*x
%%
% * (e) 
x+y
%%
% * (f) 
x'*y
%%
% * (g) The inner product you find should return a scalar value 0, what does this mean about
% these two vectors?
% they are perpendicular to each other
%%
% * (h) 
z=x.*y
% It prevents the products from being added together
 
%% 3. Plotting
%%
% * (a) 
a = rand(2);
b = rand(2);
plot([0, a(1)], [0, a(2)], 'r'); hold on
plot([0, b(1)], [0, b(2)], 'b');
xlim([0,1]);ylim([0,1])
title('Sample two vector plot')

% 4. Submission
% (a) Save the file with the code from problems 2 and 3. Go to the top. Any answers t

