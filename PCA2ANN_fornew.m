clc;
clear all;

% Load new data
load data20200730.mat

% Load trained networks
button =0;  %defualt: 0
q = 0.95;     %defualt: 95%
if button
    name=strcat('cN_pca',num2str(q*100),'_networks.mat');
else
    name=strcat('pca',num2str(q*100),'_networks.mat');
end
load(name)

% import combined features after PCA
k= net.inputs{1}.size;
if button
    k=k-1;
end
fdata=zscore(data)*coeff(:,1:k);
fdata=(fdata-repmat(min(fdata,[],1),size(fdata,1),1))...
    *spdiags(1./(max(fdata,[],1)-min(fdata,[],1))',0,size(fdata,2),size(fdata,2));
if button
   cN_r=mapminmax(cN',0.0001,0.9999)';
   fdata=[fdata,cN_r];
end


% Prediction from the Networks
x = fdata';
y = net(x);


% Gnerate Figures from old dataset
t = pN';
figure, plotconfusion(t,y)
title('All')
figure, plotroc(t,y)

fdata_cN0=fdata(cN<1,:);
x = fdata_cN0';
t = pN_cN0';
y_cN0=net(x);
figure;plotconfusion(t,y_cN0)
title('cN0')

fdata_T12cN0=fdata(cN<1&cT<3,:);
x = fdata_T12cN0';
t = pN_T12cN0';
y_T12cN0=net(x);
figure;plotconfusion(t,y_T12cN0)
title('T12+cN0')

