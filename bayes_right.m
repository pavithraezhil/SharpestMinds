clear all;
clc;
load('features.mat')
mu=0;
likelihood=0;
%calculate mean and standard deviation 
 for k=1:size(lbp_train,1)
      lbp_train_mean1=mean(lbp_train{k,1},2);
      lbp_train_mean2=mean(lbp_train{k,2},2);
      lbp_train_mean{k}=(lbp_train_mean1+lbp_train_mean2)/2;
      lbp_train_sd1=cov(lbp_train{k,1});
      lbp_train_sd2=cov(lbp_train{k,2});
      lbp_train_sd{k}=(lbp_train_sd1+lbp_train_sd2)/2;
 end
%Bayes frmula
 for d=1:size(lbp_test,2)
   for s=1:size(lbp_test{1,1},2)
       x=(lbp_test{1,d}(s));
       mu=lbp_train_mean(d);
       exponent=-(((x)-cell2mat(mu)).*2)/(2*(lbp_train_sd{1,d}.*2));
     gauss_prob(s)=(1/(sqrt(2*pi*lbp_train_sd{1,d}))*exponent);
     likelihood=likelihood+log10(gauss_prob(s));
   end
   likelihood_prob(d)=likelihood;
   [M,I]=max(likelihood_prob);
   predicted(d)=I; %storing index of the max value as predicted
   likelihood=0;
 end
%calculate accuracy
misclassification=0;
figure();
hold on;
for z=1:200
    if(predicted(z)~=z)
       misclassification=misclassification+1;
    end
    stem(z,predicted(z));
end
hold off;
title('Bayes');
xlabel('Actual value index');
ylabel('Predicted value index');
accuracy=((200-misclassification)/200)*100;
    
        