load('features.mat')
% cov of the whole training data
likelihood=0;
for k=1:size(lbp_train,1)
      lbp_train_sd1=cov(lbp_train{k,1});
      lbp_train_sd2=cov(lbp_train{k,2});
      lbp_train_sd{k}=(lbp_train_sd1+lbp_train_sd2)/2;
end
 
 for d=1:size(lbp_test,2)
   for s=1:size(lbp_test{1,1},2)
       x=(lbp_test{1,d}(s));
   end
 end

for i=1:200
mean_face1 = mean(X_train1(:,:,i), 2);
mean_face2 = mean(X_train2(:,:,i), 2);
mean_face=[mean_face1+mean_face2]/2;
mean_face_cell{i}=mean_face;

shifted_images1 = X_train1(:,:,i) - repmat(mean_face, 1);
shifted_images2 = X_train2(:,:,i) - repmat(mean_face, 1);
shifted_images=[shifted_images1+shifted_images2]/2;
shifted{i}=shifted_images;

[evectors1, score1, evalues1] = princomp(X_train1(:,:,i)');
[evectors2, score2, evalues2] = princomp(X_train2(:,:,i)');
evectors=[evectors1+evectors2]/2;
num_eigenfaces = 20;
evectors = evectors(:, 1:num_eigenfaces);
evec{i}=evectors;
features = evectors' * shifted_images;
features_cell{i}=features;
features_2D=features(:)';
evec_features_mean(i)=mean(features_2D,2);

evec_features_sd(i) = cov(features_2D);

end

%bayes classifier
for p=1:200
    feature_vec =( evec{p}' * (X_test(:,:,p) - mean_face));
    feature_vec_array{p}=feature_vec;
    
    for k=1:20
       x=mean(feature_vec_array{p},2);
       mu=evec_features_mean(p);
       sd=evec_features_sd(p);
       %sd=0.5
       exponent=-((x(k)-mu).*2)/(2*(sd.*2));
       gauss_prob(k)=(1/(sqrt(2*pi*sd))*exponent);
       likelihood=likelihood+log10(gauss_prob(k));
 
%  similarity_score(k)= ( 1 / (1 + norm(features_cell{k} - feature_vec_array{k})));
%  disp(similarity_score)
% % %  
% % % find the image with the highest similarity
%  [match_score, match_ix] = max(similarity_score);
% % 
%  match_score_cell{k}=match_score;
%  match_ix_cell{k}=match_ix;
    end
   likelihood_prob(p)=likelihood;
   [M,I]=max(likelihood_prob);
   predicted(p)=I;
   likelihood=0;
end

misclassification=0;
for z=1:200
    if(predicted(z)~=z)
       misclassification=misclassification+1;
    end
end
    accuracy=((200-misclassification)/200)*100;
