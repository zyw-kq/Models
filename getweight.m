function [coeff_weight,order_p,order_n]=getweight(coeff,k,epsilon,input_weight,button)

if button
    coeff_weight=coeff(:,1:k-1)*input_weight(1:k-1); %PCAȨ��*����Ȩ�أ��۳�cN�Ĺ���
else
    coeff_weight=coeff(:,1:k)*input_weight(1:k); %PCAȨ��*����Ȩ�أ�
end

coeff_weight(abs(coeff_weight)<epsilon)=0;


coeff_weight_p=coeff_weight;
coeff_weight_p(coeff_weight<0)=0;

[sorted, indexes] = sort(coeff_weight_p(:,1), 'descend');%�ӷ�����������ʸ���ҷ���ϵ��
order_p=indexes(sorted>0);


coeff_weight_n=coeff_weight;
coeff_weight_n(coeff_weight>0)=0;

[sorted, indexes] = sort(coeff_weight_n(:,1));
order_n=indexes(sorted<0);


end