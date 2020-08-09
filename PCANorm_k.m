function [fdata,coeff,explained]=PCANorm_k(data,train_num)
%The function is to process data and give a well-processed outputs as
%training data


[coeff,scores,~,~,explained]=pca(zscore(data));%zscore��̬������

numPCs=train_num;

fdata=scores(:,1:numPCs);
fdata=(fdata-repmat(min(fdata,[],1),size(fdata,1),1))...
    *spdiags(1./(max(fdata,[],1)-min(fdata,[],1))',0,size(fdata,2),size(fdata,2));
%ʹ��֮��Ҫ�������Ļ�����

toc
disp('��������Ŀ��')
disp(train_num)
end