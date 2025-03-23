clc;
clear;
close all;
%This is data from book scott fogler chapter7 problem D28 (Third Edition) 
% داده‌ها
X = (1:14);
Y1 = [1.33e-7 5.27e-7 0.30e-7 3.78e-7 7.56e-7 10.3e-7 17e-7 38.4e-7 ...
      70.8e-7 194e-7 283e-7 279e-7 306e-7 289e-7];

x = X;
t = Y1;

% نرمال‌سازی دستی برای بهبود یادگیری
x = mapminmax(x, 0, 1);
t = mapminmax(t, 0, 1);

% انتخاب تابع آموزش
trainFcn = 'trainbr';  % Bayesian Regularization for better generalization

% ایجاد شبکه عصبی با ۱۰ نرون در لایه پنهان
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize, trainFcn);

% تنظیم توابع فعال‌سازی
net.layers{1}.transferFcn = 'tansig';  % لایه مخفی
net.layers{2}.transferFcn = 'purelin'; % لایه خروجی

% تنظیم پیش‌پردازش ورودی و خروجی
net.input.processFcns = {'removeconstantrows', 'mapminmax'};
net.output.processFcns = {'removeconstantrows', 'mapminmax'};

% روش تقسیم داده‌ها برای آموزش، اعتبارسنجی و تست
net.divideFcn = 'divideblock';  % تقسیم پیوسته داده‌ها
net.divideMode = 'sample';      
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% انتخاب تابع ارزیابی عملکرد
net.performFcn = 'mse';  % Mean Squared Error

% تنظیم تعداد epochs برای بهبود یادگیری
net.trainParam.epochs = 500;
net.trainParam.show = 10;

% آموزش شبکه
[net, tr] = train(net, x, t);

% تست شبکه
y = net(x);
e = gsubtract(t, y);
performance = perform(net, t, y);

% نمایش عملکرد در مجموعه‌های مختلف
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net, trainTargets, y);
valPerformance = perform(net, valTargets, y);
testPerformance = perform(net, testTargets, y);

% نمایش شبکه
view(net);

% رسم نمودارها
figure, plotfit(net, x, t);
figure, plotperform(tr);
figure, plotregression(t, y);
