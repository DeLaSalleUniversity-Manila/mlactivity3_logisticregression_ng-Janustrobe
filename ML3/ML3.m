clear all; close all; clc

x = load('ex4x.dat'); 
y = load('ex4y.dat');

[m, n] = size(x);

% intercept term to x
x = [ones(m, 1), x]; 

figure
pos = find(y); neg = find(y == 0);
plot(x(pos, 2), x(pos,3), '+')
hold on
plot(x(neg, 2), x(neg, 3), 'o')
hold on
xlabel('Exam 1 score')
ylabel('Exam 2 score')

theta = zeros(n+1, 1);

g = inline('1.0 ./ (1.0 + exp(-z))'); 

% Newton's method
MAX_ITR = 7;
J = zeros(MAX_ITR, 1);

for i = 1:MAX_ITR
    z = x * theta;
    h = g(z);

    grad = (1/m).*x' * (h-y);
    H = (1/m).*x' * diag(h) * diag(1-h) * x;
    
    J(i) =(1/m)*sum(-y.*log(h) - (1-y).*log(1-h));
    
    theta = theta - H\grad;
end
theta

prob = 1 - g([1, 20, 80]*theta)

plot_x = [min(x(:,2))-2,  max(x(:,2))+2];
plot_y = (-1./theta(3)).*(theta(2).*plot_x +theta(1));
plot(plot_x, plot_y)
legend('Admitted', 'Not admitted', 'Decision Boundary')
hold off

figure
plot(0:MAX_ITR-1, J, 'o--', 'MarkerFaceColor', 'r', 'MarkerSize', 8)
xlabel('Iteration'); ylabel('J')
J