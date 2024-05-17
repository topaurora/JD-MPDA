%% JD-MPDA in setting 3 Batch Slave1 as target domain
clc
clear
load data_CQU
XS = master(:,2:7);
XS = XS';
ys = master(:,1);
XT = slave2(:,2:7);
yt = slave2(:,1);
XT =XT';

% parameter
d = 6;
sig = 80;
yit = 10;
gam = 20;
kc = 8;
kp = 70;
kt = 60;
yib = 0.5;
bestc = 1;
bestg = 0.1667;
accuracy = 0;

%% Model
[Gc,Gp,Gt] = calculation_MPDA(XS,XT,ys,kc,kp,kt,10);

    CDD = zeros(6,1);
for r = 1:100

    [XS_new,XT_new] = reduction(XS,XT,CDD,d,sig,yit,yib,gam,Gc,Gp,Gt);
    
    data = zscore(XS_new');
    
    data2 = zscore(XT_new');
    
    X_train = data;
    
    X_test = data2;
    
    y_train = ys;
    
    y_test = yt;

    
%% SVM        
    
    cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg)];

    
    model = svmtrain(y_train,X_train,cmd);
    
    [predict_label,acc,~] = svmpredict(y_test,X_test,model);
    
    acc_test = length(find(predict_label == y_test))/length(y_test);  % accuracy is determined by dividing the number of correctly predicted labels by the total number.
    
    clc
    
    if accuracy < acc_test
    
        accuracy = acc_test;

    end
    
    [distance] = iteration_CQU_CDD(XS,XT,ys,predict_label);
    
    CDD = distance;
    
end
    
clc
disp(['Source domain: Batch Master',', Target domain: Batch Slave ',num2str(2),', ','Accuracy = ' num2str(accuracy*100,'%.2f'),'%']);
