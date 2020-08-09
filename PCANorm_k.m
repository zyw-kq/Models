function [fdata,coeff,explained]=PCANorm_k(data,train_num)
%The function is to process data and give a well-processed outputs as
%training data


[coeff,scores,~,~,explained]=pca(zscore(data));%zscore正态化处理

numPCs=train_num;

fdata=scores(:,1:numPCs);
fdata=(fdata-repmat(min(fdata,[],1),size(fdata,1),1))...
    *spdiags(1./(max(fdata,[],1)-min(fdata,[],1))',0,size(fdata,2),size(fdata,2));
%使用之后要进行中心化操作

toc
disp('新特征数目：')
disp(train_num)
end