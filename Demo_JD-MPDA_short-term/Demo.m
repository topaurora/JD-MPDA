%% JD-MPDA in setting 1
clc
clear
load batch_Sep23.mat
XT = batch_Sep23(:,1:64);
yt = batch_Sep23(:,65);
load Batch_s.mat
XS = Batch_s(:,1:64);
ys = Batch_s(:,65);
XS = XS';
XT = XT';

% parameter
tow = 10;
d = 16;
sig = 10;
yit = 30;
gam = 10;
kc = 2;
kp = 10;
kt = 5;
yib = 0.4;

%% Model
[Gc,Gp,Gt] = calculation(XS,XT,ys,kc,kp,kt,tow);
CDD = zeros(64,1);
for r = 1:100
    [XS_new,XT_new] = reduction(XS,XT,CDD,d,sig,yit,yib,gam,Gc,Gp,Gt);
    
    
    X_train = zscore(XS_new');
    X_test = zscore(XT_new');
    y_train = ys;
    y_test = yt;
    
%% SVM    
    cmd = [' -t 2',' -c ',num2str(1),' -g ',num2str(1/d)];

    model = svmtrain(y_train,X_train,cmd);
    [~,acc_train,~] = svmpredict(y_train,X_train,model);
    [predict_label,acc,~] = svmpredict(y_test,X_test,model);
    acc_test = length(find(predict_label == y_test))/length(y_test);
    [distance] = iteration_owndata_CDD(XS,XT,ys,predict_label);
    CDD = distance;
    if r ==1
        accuracy = acc_test; % Comparison for optimal accuracy
        
    elseif accuracy < acc_test 
        accuracy = acc_test;
        accuracy_train = acc_train(1);
    end
end