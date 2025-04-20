% =========================================================================
%  Self-Organizing Map (SOM) Neural Network
%  Author      : Mohammad Dehbozorgi (MO.DBZ)
%  Created on  : 1404/01/31 (Persian Calendar)
%  Updated on  : 2025/04/20
%  Description : Competitive Learning using SOM Neural Network
% =========================================================================
%  License:
%  This code is provided as-is without any warranty. 
%  You may use, modify, and distribute it for educational 
%  and research purposes with proper credit to the author.
% =========================================================================

%% Initialization
clc;
close all;
clear;

%% Load Dataset
% 'simpleclusterInputs' is loaded from MATLAB built-in dataset
load simplecluster_dataset
inputs = simpleclusterInputs;

%% SOM Network Configuration
latticeSize   = [10 10];        % Grid size: 10x10 neurons
coverSteps    = 20;             % Number of steps for initial neighborhood
initNeighbor  = 3;              % Initial neighborhood size
topologyFcn   = 'gridtop';      % Grid topology function
distanceFcn   = 'linkdist';     % Distance function

% Create SOM Network
net = selforgmap(latticeSize, coverSteps, initNeighbor, topologyFcn, distanceFcn);

% Training Parameters
net.trainParam.showWindow     = true;
net.trainParam.showCommandLine = false;
net.trainParam.show           = 1;
net.trainParam.epochs         = 1000;

% Define Plot Functions
net.plotFcns = { ...
    'plotsomtop',    ... % Topology
    'plotsomnc',     ... % Neighbor connections
    'plotsomnd',     ... % Neighbor distances
    'plotsomplanes', ... % Weight planes
    'plotsomhits',   ... % Neuron hits
    'plotsompos'     ... % Neuron positions
};

%% Train SOM Network
[net, tr] = train(net, inputs);

%% Test Network
outputs = net(inputs);

%% Visualize SOM Network
figure, plotsomtop(net);
figure, plotsomnc(net);
figure, plotsomnd(net);
figure, plotsomplanes(net);
figure, plotsomhits(net, inputs);
figure, plotsompos(net, inputs);
view(net);

%% Deployment Options (set to true if needed)
% Generate function for deployment
if false
    genFunction(net, 'myNeuralNetworkFunction');
    outputs = myNeuralNetworkFunction(inputs);
end

% Generate matrix-only function for MATLAB Coder
if false
    genFunction(net, 'myNeuralNetworkFunction', 'MatrixOnly', 'yes');
    outputs = myNeuralNetworkFunction(inputs);
end

% Generate Simulink model
if false
    gensim(net);
end
