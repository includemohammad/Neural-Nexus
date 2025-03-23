clc;
clear;
close all;
%This is data from book scott fogler chapter7 problem D28 (Third Edition) 
% داده‌ها
X = (1:14)';

Y1 = [1.33e-7 5.27e-7 0.30e-7 3.78e-7 7.56e-7 10.3e-7 17e-7 38.4e-7 ...
      70.8e-7 194e-7 283e-7 279e-7 306e-7 289e-7]';

Y2 = [2e-7 6.71e-7 1.11e-7 5.72e-7 3.71e-7 8.32e-7 21.1e-7 ...
      37.6e-7 74.2e-7 180e-7 269e-7 237e-7  256e-7 149e-7]';

Y3 = [2e-7 3.78e-7 5.79e-7 0 9.36e-7 6.68e-7 17.6e-7 35.5e-7 66.1e-7 ...
      143e-7 160e-7 170e-7 165e-7 163e-7]';

Y4 = [3e-7 4.16e-7 5.34e-7 7.35e-7 6.01e-7 8.61e-7 10.1e-7 18.8e-7 ...
      28.9e-7 36.2e-7 42.2e-7 44.2e-7 46.9e-7 46.9e-7]';

T = [Y1, Y2, Y3, Y4]; % تجمیع خروجی‌ها در یک ماتریس

% نرمال‌سازی داده‌ها
[Xn, Xsettings] = mapminmax(X', 0, 1);
[Tn, Tsettings] = mapminmax(T', 0, 1);

% انتخاب تابع آموزش
trainFcn = 'trainbr';  % Bayesian Regularization

% ایجاد شبکه عصبی با ۱۰ نرون در لایه پنهان
hiddenLayerSize = [20 5];
net = fitnet(hiddenLayerSize, trainFcn);

% تنظیم توابع فعال‌سازی
net.layers{1}.transferFcn = 'tansig';  % لایه مخفی
net.layers{2}.transferFcn = 'tansig'; 
net.layers{3}.transferFcn = 'purelin'; % لایه خروجی

% تنظیم پیش‌پردازش ورودی و خروجی
net.input.processFcns = {'removeconstantrows', 'mapminmax'};
net.output.processFcns = {'removeconstantrows', 'mapminmax'};

% روش تقسیم داده‌ها
net.divideFcn = 'divideblock';  % تقسیم پیوسته داده‌ها
net.divideMode = 'sample';      
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% انتخاب تابع ارزیابی عملکرد
net.performFcn = 'mse';

% تنظیم تعداد epochs
net.trainParam.epochs = 500;
net.trainParam.show = 10;

% آموزش شبکه
[net, tr] = train(net, Xn, Tn);

% تست شبکه
Ypred = net(Xn);
e = gsubtract(Tn, Ypred);
performance = perform(net, Tn, Ypred);

% نمایش عملکرد
trainTargets = Tn .* tr.trainMask{1};
valTargets = Tn .* tr.valMask{1};
testTargets = Tn .* tr.testMask{1};
trainPerformance = perform(net, trainTargets, Ypred);
valPerformance = perform(net, valTargets, Ypred);
testPerformance = perform(net, testTargets, Ypred);

% نمایش شبکه
view(net);

% رسم نمودارها
figure, plotperform(tr);
figure, plotregression(Tn, Ypred);

% نمایش خروجی‌های واقعی و پیش‌بینی‌شده
Ypred_real = mapminmax('reverse', Ypred, Tsettings);
figure, plot(X, T, 'o', X, Ypred_real', '-');
legend('Y1','Y2','Y3','Y4','Y1 Pred','Y2 Pred','Y3 Pred','Y4 Pred');
xlabel('X'); ylabel('Y values');
title('Actual vs. Predicted Outputs');
