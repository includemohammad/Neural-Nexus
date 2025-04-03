% ========================================================================
%  MATLAB Code
%  Author      : Mohammad Dehbozorgi (MO.DBZ)
%  Created on  : [1404/01/14]
%  Description : [This is Code for SVR Method in Neural Network ]

% ========================================================================
%  License:
%  This code is provided as-is without any warranty. 
%  You are free to use, modify, and distribute it for educational 
%  and research purposes, provided that proper credit is given to 
%  the original author: Mohammad Dehbozorgi (MO.DBZ).
% ========================================================================
clc;
clear;
close all;

%% Load Data 
x = linspace(0, 2*pi, 40);  % Generate input data
t = 2*x + 1 + 0.7*randn(size(x));  % Generate target data with noise
n = numel(t);  % Number of data points

% Plot the input data
figure;
plot(x, t, 'bo', 'MarkerSize', 6, 'LineWidth', 1.2); 
xlabel('x');
ylabel('t');
title('SVR with Epsilon-Tube');
hold on;

%% Design SVR 
epsilon = 1;  % Epsilon value for SVR
C = 1;  % Regularization parameter

% Construct the Hessian matrix H
H = zeros(n, n);
for i = 1:n
    for j = i:n
        H(i, j) = x(:, i)' * x(:, j);
        H(j, i) = H(i, j);  % Ensure symmetry
    end
end 

% Construct the full Hessian matrix for quadratic programming
HH = [H -H; -H H];
HH = HH + 1e-6 * eye(size(HH));  % Ensure positive semi-definiteness

% Define optimization parameters
f = [-t'; t'] + epsilon;
Aeq = [ones(1, n) -ones(1, n)];
beq = 0;
Ib = zeros(2*n, 1);
ub = C * ones(2*n, 1);

% Set optimization options
options = optimset('Display', 'Iter', 'MaxIter', 100);

% Solve quadratic programming problem
Alpha = quadprog(HH, f, [], [], Aeq, beq, Ib, ub, [], options);

%% Extract Support Vectors
alpha_plus = Alpha(1:n);
alpha_minus = Alpha(n+1:end);

% Find indices of support vectors (0 < alpha < C)
sv_indices = find((alpha_plus > 1e-6 & alpha_plus < C) | (alpha_minus > 1e-6 & alpha_minus < C));

% Extract support vectors
x_sv = x(sv_indices);
t_sv = t(sv_indices);

%% Compute Weights and Bias (Fixed)
% Compute weight vector (w) using all alphas
w = sum((alpha_plus - alpha_minus)' .* x);

% Compute bias (b) using support vectors
b_values = t_sv - w * x_sv;
b = mean(b_values);

% Display results
disp(['Corrected Weight (w): ', num2str(w)]);
disp(['Corrected Bias (b): ', num2str(b)]);

%% Plot SVR Regression Line and Epsilon-Tube
x_fit = linspace(min(x), max(x), 100);
y_fit = w * x_fit + b;

% Plot SVR regression line
plot(x_fit, y_fit, 'r-', 'LineWidth', 2);

% Plot the epsilon-tube (margin)
plot(x_fit, y_fit + epsilon, 'k--', 'LineWidth', 1.5);  % Upper epsilon bound
plot(x_fit, y_fit - epsilon, 'k--', 'LineWidth', 1.5);  % Lower epsilon bound

% Highlight support vectors
plot(x_sv, t_sv, 'rs', 'MarkerSize', 10, 'LineWidth', 2);

legend('Data Points', 'SVR Regression Line', 'Epsilon Boundaries', 'Support Vectors');
hold off;




