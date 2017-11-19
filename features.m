load('data.mat')
mu=0;
k=1;
%Pre-processing
for i=1:size(face,3)
   face(:,:,i)=adapthisteq(face(:,:,i));
   face(:,:,i)=imsharpen(face(:,:,i));
   face(:,:,i)=imadjust(face(:,:,i));
end
% Y has class labels
for i=1:size(face,3)/3
    for j=1:2
        Y_train(k)=i;
        Y_test(i)=i;
        k=k+1;
    end
end
num_points=600;
X_test =(face(:,:,(3:3:num_points)));
X_train1=(face(:,:,(1:3:num_points)));
X_train2=(face(:,:,(2:3:num_points)));
for g=1:size(X_train1,3)
    X_try=cat(3,X_train1(:,:,g),X_train2(:,:,g));
    X_train{g}=X_try;
end

clear face, num_points;

%LBP of training data
    for i=1:size(X_train,2)
        for b=1:2
        lbp = extractLBPFeatures(X_train{1,i}(:,:,b),'Upright',true);
           for v=1:size(lbp,2)
            mu=mu+lbp(v);
           end
           mu=mu/size(lbp,2);
           for j=1:size(lbp,2)
              lbp(j)=i*i*i*i*i*(log10(lbp(j)+mu));
           end
           mu=0;
           lbp_train{i,b}=lbp;
           %adding mean lbp value to prevent multiplication with zero
           %taking log to prevent misclassificatin due to floating point
           %errors arising out of small values
        end
    end
    %LBP of test data
    mu=0;
    for i=1:size(X_test,3)
        lbp = extractLBPFeatures(X_test(:,:,i),'Upright',true);
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

clear v,j,k,mu,lbp
    save('features.mat','lbp_train','lbp_test','X_train','X_train1','X_train2','Y_train','X_test','Y_test')
   