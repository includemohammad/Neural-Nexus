% ========================================================================
%  MATLAB Code
%  Author      : Mohammad Dehbozorgi (MO.DBZ)
%  Created on  : [1404/01/17]
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

%% Load data
S = load('MyData1.mat');
varNames = fieldnames(S);
x = S.(varNames{1});  % Load first variable

%% Shape fix
if isvector(x)
    x = x(:);  % Make it a column vector if needed
end

%% Run K-means
nClusters = 3;
DistanceMetric = 'cityblock';  % Can be 'sqEuclidean', 'cityblock', etc.
Options = statset('Display', 'final');  % Corrected typo from 'Disply' to 'Display'
[I, C, D] = kmeans(x, nClusters, 'Distance', DistanceMetric, 'Options', Options);

%% Plot
figure;
hold on;
axis equal;

colors = lines(nClusters);  % Generate distinct colors

if size(x,2) == 1
    % 1D case
    scatter(x, zeros(size(x)), 50, I, 'filled');

    % Plot centers
    scatter(C, zeros(size(C)), 100, 'kx', 'LineWidth', 2);

    % Draw circles around clusters
    for k = 1:nClusters
        members = x(I == k);
        radius = mean(abs(members - C(k)));  % average distance from center
        theta = linspace(0, 2*pi, 100);
        circleX = C(k) + radius * cos(theta);
        circleY = radius * sin(theta);
        plot(circleX, circleY, 'Color', colors(k,:), 'LineWidth', 1.5);
    end

    xlabel('Feature');
    title('K-means Clustering with Centers & Circles (1D)');
else
    % 2D or more
    gscatter(x(:,1), x(:,2), I, [], 'o', 8);
    scatter(C(:,1), C(:,2), 100, 'kx', 'LineWidth', 2);  % Centers

    % Draw circle per cluster
    for k = 1:nClusters
        members = x(I == k, :);
        dists = vecnorm(members - C(k,:), 2, 2);  % Euclidean distances
        radius = mean(dists);  % Approx. average radius
        theta = linspace(0, 2*pi, 100);
        circleX = C(k,1) + radius * cos(theta);
        circleY = C(k,2) + radius * sin(theta);
        plot(circleX, circleY, 'Color', colors(k,:), 'LineWidth', 1.5);
    end

    xlabel('Feature 1');
    ylabel('Feature 2');
    title('K-means Clustering with Centers & Circles (2D)');
end

legend('off');
grid on;


