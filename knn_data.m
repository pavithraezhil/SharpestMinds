clear all;
load('features.mat');
k=7;

for i=1:size(lbp_test,2)
    for f=1:size(lbp_train{1,1},2)
        for j=1:size(lbp_train,1)
           dist1(j)=sqrt((log10(lbp_train{j,1}(f))-log10(lbp_test{1,i}(f))).^2);
           dist2(j)=sqrt((log10(lbp_train{j,2}(f))-log10(lbp_test{1,i}(f))).^2);
        end
        a3=[dist1; dist2];
        
        a3=a3(:)';
        [a3,Ind]=sort(a3);
       %Ind=(Ind+mod(Ind,2))/2;
        a3_Ind=Ind(1:k);
        a=Y_train(a3_Ind);
        majority=0;
        for e=1:k
            key=a(e); 
            for q=e+1:k
                if(key==a(q))
                    majority=majority+1;
                    a3_mode=key;
                    break;
                end
            end
            if(majority>0)
                break;
            end
        end
        if(majority==0)
            a3_mode=a3_Ind(1);
        end
        index(f)=a3_mode;
    end
    M=mode(index);
    predicted(i)=M;
end
%calculate accuracy
misclassification=0;
for z=1:200
    if(predicted(z)~=z)
       misclassification=misclassification+1;
    end
end
    accuracy=((200-misclassification)/200)*100;

