
% =========================================================================
%  Self-Organizing Map (SOM) Neural Network - PCA Visualization
%  Author      : Mohammad Dehbozorgi (MO.DBZ)
%  Created on  : 1404/02/3 (Persian Calendar)
%  Updated on  : 2025/04/23
%  Description :  Competitive Learning PCA 
% =========================================================================
%  License:
%  This code is provided as-is without any warranty. 
%  You may use, modify, and distribute it for educational 
%  and research purposes with proper credit to the author.
% =========================================================================

clc; 
clear all;
close all

%% Load Data
N1 = 300;
MU1 = [1 2];
SIGMA1 = [5 2; 2 2];
X1 = mvnrnd(MU1, SIGMA1, N1)';
T1 = 1 * ones(1, N1);

N2 = 500;
MU2 = [3 6];
SIGMA2 = [5 0.4; 0.4 2];
X2 = mvnrnd(MU2, SIGMA2, N2)';
T2 = 2 * ones(1, N2);

X = [X1 X2];
T = [T1 T2];

%% Perform PCA manually
X_mean = mean(X, 2);                              % Mean along rows (each feature)
X_centered = X - repmat(X_mean, 1, size(X, 2));   % Subtract mean from each column

C = cov(X_centered');                             % Covariance matrix
[Q, D] = eig(C);                                  % Eigen decomposition
lambda = diag(D);                                 % Eigenvalues

% Sort eigenvalues and eigenvectors
[lambda, idx] = sort(lambda, 'descend');
Q = Q(:, idx);                                    % Principal directions sorted

% Project data onto the principal components
Y = Q' * X_centered;                              % 2 × N matrix: projected data

%% Plot original data with PCA vectors
figure(1)
hold on
grid on
axis equal
title('PCA Vectors with Original Data')
xlabel('X_1')
ylabel('X_2')

% Plot data points
plot(X(1,T==1), X(2,T==1), 'ro')
plot(X(1,T==2), X(2,T==2), 'bs')

% Plot mean point
m = X_mean;
plot(m(1), m(2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 8)

% Normalize and plot eigenvectors
scale = 3 * sqrt(lambda);  % scale factor

for i = 1:2
    vec = scale(i) * Q(:, i);
    quiver(m(1), m(2), vec(1), vec(2), 0, 'k', 'LineWidth', 2, 'MaxHeadSize', 2)
    text(m(1)+vec(1)*1.1, m(2)+vec(2)*1.1, sprintf('PC%d', i), ...
         'FontSize', 12, 'Color', 'k', 'FontWeight', 'bold')
end

%% Plot PCA-projected data (features Y1 and Y2)
figure(2)

subplot(2,2,1);
plot(Y(1,T==1), Y(2,T==1), 'ro');
hold on;
plot(Y(1,T==2), Y(2,T==2), 'bs');
legend('Class A', 'Class B');
title('Projected Data: Y_1 vs Y_2');
xlabel('PC1'); ylabel('PC2');
grid on;

subplot(2,2,2);
plot(Y(1,T==1), 'r-o');
hold on;
plot(Y(1,T==2), 'b-s');
legend('Class A', 'Class B');
title('Feature Y_1 (PC1)');
xlabel('Sample Index'); ylabel('Value');
grid on;

subplot(2,2,3);
plot(Y(2,T==1), 'r-o');
hold on;
plot(Y(2,T==2), 'b-s');
legend('Class A', 'Class B');
title('Feature Y_2 (PC2)');
xlabel('Sample Index'); ylabel('Value');
grid on;

