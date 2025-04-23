% =========================================================================
%  Self-Organizing Map (SOM) Neural Network
%  Author      : Mohammad Dehbozorgi (MO.DBZ)
%  Created on  : 1404/02/3 (Persian Calendar)
%  Updated on  : 2025/04/23
%  Description : Competitive Learning PCA 
% =========================================================================
%  License:
%  This code is provided as-is without any warranty. 
%  You may use, modify, and distribute it for educational 
%  and research purposes with proper credit to the author.
% =========================================================================

clc;
clear all
close all 

%% Load Data

MU = [1 2];
SIGMA =[5  2; 
       2   2]; % This matrix must have one subdiagonal.
X = mvnrnd(MU,SIGMA,10000)';
%% Perform PCA  

C = cov(X');

[Q LANDA] = eig (C) ;

lambda = diag (LANDA) ; 

[lambda  SortOrder ] = sort(lambda,'descend') ;

LANDA = diag (lambda);
Q = Q (:,SortOrder);
%% Plot 
figure (1)
plot(X(1,:),X(2,:),'o');
hold on 
lq1 = lambda(1)*Q(:,1);
plot([MU(1) MU(1)+lq1(1)],[MU(2) ,MU(2)+lq1(2)],'r','LineWidth',2);
lq2 = lambda(2)*Q(:,2);
plot([MU(1) MU(1)+lq2(1)],[MU(2) ,MU(2)+lq2(2)],'r','LineWidth',2);
grid on ;
axis equal;

figure (2) 

subplot (1,2,1)
q1 = Q(:,1);
Y1 = q1'*X;
q2 = Q(:,2);
Y2 = q2'*X;
plot(Y1,'b');
hold on 
plot(Y2,'r');