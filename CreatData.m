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
function x = CreatData(m)
    xA1 = 10; %this is lpocation of clusturing Data 1
    xA2 = 10;
    xA = [xA1+randn(m,1) xA2+randn(m,1)];
    
    xB1 = 50; %this is lpocation of clusturing Data 2
    xB2 = 80;
    xB = [xB1+randn(m,1) xB2+randn(m,1)];
    
    xC1 = 10; %this is lpocation of clusturing Data 3
    xC2 = 30;
    xC = [xC1+randn(m,1) xC2+randn(m,1)];
    
    x=[xA;xB;xC];
    plot(x(:,1),x(:,2),'o');
end
