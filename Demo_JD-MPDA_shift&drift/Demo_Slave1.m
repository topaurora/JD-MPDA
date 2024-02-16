%% JD-MPDA in setting 3 Batch Slave1 as target domain
clc
clear
load data_CQU
XS = master(:,2:7);
XS = XS';
ys = master(:,1);
XT = slave1(:,2:7);
yt = slave1(:,1);
XT =XT';
% parameter 
d = 6;
sig = 80;
yit = 10;
gam = 10;
kc = 32;
kp = 70;
kt = 50;
accuracy = 0;
yib = 0.6;

%% Model
% Calculate the portion of the iteration that does not change
[Gc,Gp,Gt] = calculation(XS,XT,ys,kc,kp,kt,10);

    CDD = zeros(6,1);
    for r = 1:100 % the cycle
        [XS_new,XT_new] = reduction(XS,XT,CDD,d,sig,yit,yib,gam,Gc,Gp,Gt); % get subspace data
        X_train = zscore(XS_new');
        X_test = zscore(XT_new');
        y_train = ys;
        y_test = yt;

%% SVM
        cmd = [' -t 2',' -c ',num2str(1),' -g ',num2str(1/d)];

        model = svmtrain(y_train,X_train,cmd);
        [predict_label,acc,~] = svmpredict(y_test,X_test,model);
        acc_test = length(find(predict_label == y_test))/length(y_test);
        if accuracy < acc_test
            accuracy = acc_test;

        end
% Update the conditional distribution distance
        [distance] = iteration_CQU_CDD(XS,XT,ys,predict_label); 
        CDD = distance;
    end
    
