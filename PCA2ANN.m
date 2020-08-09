% Generate the tongue cancer classifier with a Neural Network
% Created by Lingxiao @20200730
clc;
clear all;

% PCA parameters q-percent information storage£¬default: 95%
q = 0.99;
% import cN or not, defult: 0
button=0;

% Step0.  Load the processed data for training
load data20200730.mat
target=pN;


% Step1. PCA dimension reduction 
[fdata,coeff,explained]=PCANorm(data,q); 
if button
   cN_r = mapminmax(cN',0.0001,0.9999)';
   fdata=[fdata,cN_r];
end

% Step2. Train ANN£¬Training:Test:Validation=6:2:2
ANNClassifier
fnum=size(fdata,2);
fdata_cN0=fdata(cN<1,:);

% 2.0-a cN0 dataset
n=2;
while performance>0.4 && n<200
   disp(['Round ',num2str(n), ' ANN performence in cN0 dataset£º'])
   disp(performance)
    ANNClassifier_cN0
    n=n+1;
end

% 2.0-b T12cN0 dataset
fdata_T12cN0=fdata(cN<1&cT<3,:);
while performance>0.35 && n<500
   disp(['Round ',num2str(n), ' ANN performence in T12cN0 dataset£º'])
   disp(performance)
   ANNClassifier_T12cN0
   n=n+1;
end

% 2.0-c All dataset
while performance>0.3 && n<1001
    disp(['Round ',num2str(n),' ANN performence in All data£º'])
    disp(performance)
    ANNClassifier_reinforcement
    n=n+1;
end
if button
    name1 = strcat('cN_pca',num2str(q*100),'_confmaps');
    name2 = strcat('cN_pca',num2str(q*100),'_roc');
else
    name1 = strcat('pca',num2str(q*100),'_confmaps');
    name2 = strcat('pca',num2str(q*100),'_roc');
end
exportconfmaps(x,net,t,tr)
title('All')
saveas(gcf,name1,'epsc')
close 
exportroc(x,net,t,tr)
title('All')
saveas(gcf,name2,'epsc')
close 


  
% 2.1 Export predictions
y = net(fdata');
Prediction=round(y');
Labels=Prediction; Labels(tr.trainInd)=2; Labels(tr.valInd)=3; Labels(tr.testInd)=4;
predict= table(cN,pN,Prediction,Labels);
if button
    name=strcat('cN_pca',num2str(q*100),'_prediction.xls');
    writetable(predict,name);
else
    name=strcat('pca',num2str(q*100),'_prediction.xls');
    writetable(predict,name);
end

% 2.2 Save net and PCA coefficient matrix
if button
    name = strcat('cN_pca',num2str(q*100),'_networks.mat');
else
    name = strcat('pca',num2str(q*100),'_networks.mat');
end
save(name,'net','coeff')

% Step3. Search the important features
% 3.1 extract weights in net
n=size(x,1);
Bias=zeros(n,11);
for k=1:n
    for i=1:11
        x_m=x;
        x_m(k,:)=x_m(k,:)*(i-1)*0.1+x_m(k,:);
        y = net(x_m);
        Bias(k,i)=sum(abs(t-y))/numel(tind);
    end
end
Bias_slope=(Bias(:,2:11)-Bias(:,1:10))/0.1;
input_weight=mean(Bias_slope,2);

% 3.2£¨option£©plot the influence of the combined features
a=0;
if a
    figure
    change_percent=linspace(0,1,11);
    for i=1:n
        plot(change_percent,Bias(i,:),'DisplayName',['Number=',num2str(i)])
        hold on
    end
    hold off
    legend
    xlabel('Change Percent')
    ylabel('Bias Value')
end

% 3.3 Return top 50 features, sort by influence
epsilon=1E-3;%threshold 
[coeff_weight,order_p,order_n]=getweight(coeff,n,epsilon,input_weight,button);

top =50;%options
Positive=order_p(1:top);Negative=order_n(1:top);
output= table(Positive,Negative);
if button
    name=strcat('cN_pca',num2str(q*100),'_ranking.xls');
    writetable(output,name);
else
    name=strcat('pca',num2str(q*100),'_ranking.xls');
    writetable(output,name);
end