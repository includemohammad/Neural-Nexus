% =========================================================================
%  MATLAB Code
%  Author      : Mohammad Dehbozorgi (MO.DBZ)
%  Created on  : [1404/01/18]
%  Description : Code for Competitive Learning (SOM-like) Neural Network
% =========================================================================
%  License:
%  This code is provided as-is without any warranty. 
%  You are free to use, modify, and distribute it for educational 
%  and research purposes, provided that proper credit is given to 
%  the original author: Mohammad Dehbozorgi (MO.DBZ).
% =========================================================================

clc;
clear;
close all;

%% Load and Prepare Data
x = iris_dataset();

%% Create and Train Competitive Layer Network
nClass = 4;
eta = 0.05;

net = competlayer(nClass, eta);
view(net);

net.trainParam.epochs = 200;
net = train(net, x);
ClusterCenter = net.IW{:};

%% Apply Network to Data
y = net(x);
ClassIndx = vec2ind(y);

%% Plot Input Data (Basic)
figure(1);
plot(x(1,:), x(2,:), 'bo');
hold on;
plot(x(3,:), x(4,:), 'ro');
title('Raw Input Data');
xlabel('Feature 1 / 3');
ylabel('Feature 2 / 4');
legend('x(1) vs x(2)', 'x(3) vs x(4)');
grid on;

%% Visualize Clustering Results with Subplots
figure(2);
Colors = hsv(nClass);
Colors = 0.5 * Colors;

c = 0;
for i = 1:4
    for j = 1:4
        c = c + 1;
        subplot(4, 4, c);
        if i ~= j
            for k = 1:nClass
                idx = find(ClassIndx == k);
                plot(x(i, idx), x(j, idx), '.', 'Color', Colors(k,:), 'MarkerSize', 10);
                hold on;
            end
            plot(ClusterCenter(:, i), ClusterCenter(:, j), 'kx', 'MarkerSize', 12, 'LineWidth', 2);
            title(sprintf('x(%d) vs x(%d)', i, j));
            xlabel(['x(', num2str(i), ')']);
            ylabel(['x(', num2str(j), ')']);
            grid on;
        else
            axis off;
        end
    end
end


