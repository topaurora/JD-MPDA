clc
clear
load Drift_Dataset;
XS = Drift_dataset{1,1};

ys = Drift_dataset{1,2};
lamda = [0.6 0.9 0.9 0.9 1 0.8 0.7 0.9 0.4];
dim  = [70 100 90 100 40 110 110 100 128];
k = 3;
q = 1;

%XS = mapminmax(XS);
%XT = mapminmax(XT);
XS = XS * diag(1./sqrt(sum(XS.^2))); %向量单位化
for i = 2:2
XT = Drift_dataset{i,1};
yt = Drift_dataset{i,2};
e = lamda(i-1);
d = dim(i-1);
XT = XT * diag(1./sqrt(sum(XT.^2)));

[P,XS_sub,XT_sub,u_sc,u_subc] = CSBD(XS,ys,XT,0,d,e,0);
%us = sum(XS,2)/size(XS,2);
%ut = sum(XT,2)/size(XT,2);
%MMD = (us-ut)*(us-ut)';

%E = (XS*XS') + (e*(XT*XT'));

%A = pinv(MMD) * E;
%[o,~] = size(A);
%[P,D] = eig(A);
%[~,Idex] = sort(diag(real(D)),'descend');
%    for j = 1:o
%        if Idex[j] == q
%            if q <= d+1
%               P_(:,q)= real(P(:,j));
%                q = q+1;
%                continue;
%            end
%        end   
%    end
%P_ =real(P(:,Idex(1:d)));
%XS_new = P_' * XS;
%XT_new = P_' * XT;

Tr1 = pca_real(XS_sub);
Tr2 = pca_real(XT_sub);
Tr1 = mapminmax(Tr1);
Tr2 = mapminmax(Tr2);

subplot(2,9,i-1);
scatter(Tr1(1,:),Tr1(2,:),'filled');
xlabel('PC1');
ylabel('PC2');
title('Batch 1');
subplot(2,9,i+8);
scatter(Tr2(1,:),Tr2(2,:),'filled');
xlabel('PC1');
ylabel('PC2');
title('Batch 2');
% predict
train_matrix =normalize(XS_sub',1);
test_matrix = normalize(XT_sub',1);
train_label = ys;
test_label = yt;

%[m, n] = size(train_matrix); %m,n为输⼊⽂本矩阵的⾏列
%eps=0.0000000001;
%train_matrix = zscore(train_matrix+eps*ones(m,n));
%[m, n] = size(test_matrix); %m,n为输⼊⽂本矩阵的⾏列
%eps=0.0000000001;
%test_matrix = zscore(test_matrix+eps*ones(m,n));
%Train_matrix = mapminmax(train_matrix);
%Test_matrix = mapminmax(test_matrix);
%train_matrix = Train_matrix';
%test_matrix = Test_matrix';
[m,~] = size(test_matrix);

acc_score = zeros(k,1);
for j =1:k
    n = 0;
    y_predict = MultiSvm(train_matrix,train_label,test_matrix);
    for w = 1:m
        if y_predict(w) == test_label(w)
            n = n+1;
        end
    end
    acc = n / m;
    disp(['准确率:',num2str(acc),' batch',num2str(i)]);
    acc_score(j) = acc;
end
end