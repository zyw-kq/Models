% Solve a Pattern Recognition Problem with a Neural Network
% Original Script generated by Neural Pattern Recognition Mode
% Modified by Lingxiao @20191230

% This script assumes these variables are defined:
%
%   features - input data.
%   target - target data.

x = fdata_T12cN0';
t = pN_T12cN0';

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
% 'traingdx' Gradient descent with momentum and adaptive learning rate backpropagation

trainFcn = 'traingdm';
% trainFcn = 'trainlm';


net.performFcn = 'crossentropy';
net.performParam.regularization = 0.1;
% net.performParam.normalization = 'none';

net.trainParam.max_fail = 100;
net.trainParam.epochs = 10000;


% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 60/100;
net.divideParam.valRatio =20/100;
net.divideParam.testRatio = 20/100;

% Train the Network
[net,~] = train(net,x,t);
net.trainParam.max_fail = 20;
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);
Bias = sum(abs(t-y))/numel(tind);


% View the Network
% view(net)
weights_i=net.IW;
weights_l=net.LW;
bias=net.b;

% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, ploterrhist(e)



