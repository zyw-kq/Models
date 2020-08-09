function [coeff_weight,order_p,order_n]=getweight(coeff,k,epsilon,input_weight,button)

if button
    coeff_weight=coeff(:,1:k-1)*input_weight(1:k-1); %PCA权重*网络权重；扣除cN的贡献
else
    coeff_weight=coeff(:,1:k)*input_weight(1:k); %PCA权重*网络权重；
end

coeff_weight(abs(coeff_weight)<epsilon)=0;


coeff_weight_p=coeff_weight;
coeff_weight_p(coeff_weight<0)=0;

[sorted, indexes] = sort(coeff_weight_p(:,1), 'descend');%从方差最大的特征矢量找非零系数
order_p=indexes(sorted>0);


coeff_weight_n=coeff_weight;
coeff_weight_n(coeff_weight>0)=0;

[sorted, indexes] = sort(coeff_weight_n(:,1));
order_n=indexes(sorted<0);


end