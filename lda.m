clear all;
load('features.mat');

likelihood=0;
for k=1:size(lbp_train,1)
      lbp_train_sd1=cov(lbp_train{k,1});
      lbp_train_sd2=cov(lbp_train{k,2});
      lbp_train_sd{k}=(lbp_train_sd1+lbp_train_sd2)/2;
end
mu=0;
for i=1:200
mean_face1 = mean(X_train1(:,:,i), 3);
mean_face2 = mean(X_train2(:,:,i), 3);
mean_face=[mean_face1+mean_face2]/2;
mean_face_cell{i}=mean_face;
mean_1D=reshape(mean_face',1,504)
mu=mu+mean(mean_face,2);

shifted_images1 = X_train1(:,:,i) - repmat(mean_face, 1);
shifted_images2 = X_train2(:,:,i) - repmat(mean_face, 1);
shifted_images=[shifted_images1+shifted_images2]/2;
%shifted_images=cat(3,shifted_images1,shifted_images2);
shifted{i}=shifted_images;
end
sw=eye(size(shifted{1},2));
% Calculate the within class variance (SW)
 for f=1:200
   s1=shifted{f}'*shifted{f};
   sw=sw+s1; 
 
   invsw=inv(sw);
   invsw_cell{f}=invsw;
 end
   
   % if more than 2 classes calculate between class variance (SB)
   SB=eye(size(shifted{1},2));
   for b=1:200
      sb1=200*(mean_face_cell{b}-mu)'*(mean_face_cell{b}-mu);
      SB=SB+sb1;
      v=(invsw*SB);
      v_cell{b}=v;
   end   
   v=(invsw*SB);


 % find eigen values and eigen vectors of the (v)
 for p=1:200
     [evectors, score, evalues] = princomp(v');
     evectors=evectors(:,1:19);
     evectors_cell{p}=evectors;
     evec{i}=evectors;
     lda_test=X_test(:,:,p)*evec{i};
     lda_test_cell{p}=lda_test;
 end
 
 
 %bayes
 for k=1:size(lbp_train,1)
      lbp_train_mean1=mean(lbp_train{k,1},2);
      lbp_train_mean2=mean(lbp_train{k,2},2);
      lbp_train_mean{k}=(lbp_train_mean1+lbp_train_mean2)/2;
      lbp_train_sd1=cov(lbp_train{k,1});
      lbp_train_sd2=cov(lbp_train{k,2});
      lbp_train_sd{k}=(lbp_train_sd1+lbp_train_sd2)/2;
     end
 
 
 for d=1:size(lda_test_cell,2)
   for s=1:size(lda_test_cell{1,1},2)
       x=(lda_test_cell{1,d}(s));
       mu=lbp_train_mean(d);
       exponent=-(((x)-cell2mat(mu)).*2)/(2*(lbp_train_sd{1,d}.*2));
       gauss_prob(s)=(1/(sqrt(2*pi*lbp_train_sd{1,d}))*exponent);
       likelihood=likelihood+log10(gauss_prob(s));
   end
   likelihood_prob(d)=likelihood;
   [M,I]=max(likelihood_prob);
   predicted(d)=I;
   likelihood=0;
 end
 
 misclassification=0;
for z=1:200
    if(predicted(z)~=z)
       misclassification=misclassification+1;
    end
end
    accuracy_bayes=((200-misclassification)/200)*100;

%KNN

mu=0;
k=5;
for z=1:200
lbp = extractLBPFeatures(lda_test_cell{1,z},'Upright',true);
           for v=1:size(lbp,2)
            mu=mu+lbp(v);
           end
           mu=mu/size(lbp,2);
           for j=1:size(lbp,2)
                 lbp(j)=i*i*i*i*i*(log10(lbp(j)+mu));
           end
           mu=0;
           lbp_test{i}=lbp;
end

for i=1:size(lbp_test,2)
    for f=1:size(lbp_train{1,1},2)
        for j=1:size(lbp_train,1)
        dist1(j)=sqrt((log10(lbp_train{j,1}(f))-log10(lbp_test{1,i}(f))).^2);
        dist2(j)=sqrt((log10(lbp_train{j,2}(f))-log10(lbp_test{1,i}(f))).^2);
        
        end
        a3=[dist1; dist2];
        a3=a3(:)';
        [a3,Ind]=sort(a3);
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
    accuracy_knn=((200-misclassification)/200)*100;
