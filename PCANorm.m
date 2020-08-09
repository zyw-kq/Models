function [fdata,coeff,explained]=PCANorm(data,train_percent)
%The function is to process data and give a well-processed outputs as
%training data
    if (nargin<2)
        train_percent = 0.99;
    end
tic
num_features=size(data,2);

[coeff,scores,~,~,explained]=pca(zscore(data));%zscore正态化处理

for k=1:num_features
    per=sum(explained(1:k))/sum(explained);
    if per>train_percent 
        break
    end
end
numPCs=k;

fdata=scores(:,1:numPCs);
fdata=(fdata-repmat(min(fdata,[],1),size(fdata,1),1))...
    *spdiags(1./(max(fdata,[],1)-min(fdata,[],1))',0,size(fdata,2),size(fdata,2));
%使用之后要进行中心化操作

toc
disp('Information Storage Percent：')
disp(per)
end