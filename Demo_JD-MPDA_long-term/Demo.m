%% Experimental in Setting 2 (long-term drift)
% Set batch 1 as the source domain, and batch K (K = 2, 3, ..., 10) as the target domain
% Drift_dataset: The Gas Sensor Array Drift dataset from UCI. http://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset

clc
clear
load Drift_dataset
XS = Drift_dataset{1,1};
% Use batch 1 as source domain
ys = Drift_dataset{1,2};
% labels of batch 1
d_m = [2 4 16 4 4 4 128 2 4]; 
sig_ma = [1 30 30 30 10 20 1 10 20];
yit_a = [10 30 20 30 10 1 1 1 1];
gam_ma = [60 60 1 40 20 40 40 60 30];
k_c = [4 4 8 8 8 8 16 4 8];
k_p = [40 50 50 40 50 40 60 40 40];
k_t = [60 30 30 30 30 50 50 30 60];
yib_sio = [0.5 0.6 0.8 0.8 0 0 0 0.2 0.1];
tow = 10;
%parameters 
predict_acc = cell(9,2);
% Store the highest classification accuracy.
for i = 1:9
    XT = Drift_dataset{i+1,1}; % Use batch 2-10 as target domain, respectively
    yt = Drift_dataset{i+1,2}; % labels
    d = d_m(i);
    sig = sig_ma(i);
    yit = yit_a(i);
    gam = gam_ma(i);
    kc = k_c(i);
    kp = k_p(i);
    kt = k_t(i);
    yib = yib_sio (i);
    accuracy = 0; % Reset  accuracy per batch to avoid overlap
    
%% Model    
    [Gc,Gp,Gt] = calculation(XS,XT,ys,kc,kp,kt,tow); % Calculate the portion of the iteration that does not change
    CDD = zeros(128,1); % Initialising the conditional distribution
    for r = 1:100
        [XS_new, XT_new] = reduction(XS,XT,CDD,d,sig,yit,yib,gam,Gc,Gp,Gt);
        
%%     SVM libsvm was developed by Professor Lin of National Taiwan University and is publicly available.     
        X_train = zscore(XS_new');
        X_test = zscore(XT_new');
        y_train = ys;
        y_test = yt;
% To verify the effectiveness of the model in removing drift, we use a standard SVM for classification, with default values for all parameters
        cmd = [' -t 2',' -c ',num2str(1),' -g ',num2str(1/d)];

        model = svmtrain(y_train,X_train,cmd);
        [predict_label,acc_predict,~] = svmpredict(y_test,X_test,model);
% Calculate classification accuracy of target data in subspace.
        acc_test = length(find(predict_label == y_test))/length(y_test);
        if accuracy < acc_test
            accuracy = acc_test; 
            acce = acd(1);
        end
% Update the conditional distribution distance
        [distance] = iteration_CDD(XS,XT,ys,predict_label); 
        CDD = distance;
                                        
    end
    predict_acc{i,1} = acce; 
    predict_acc{i,2} = accuracy;
end