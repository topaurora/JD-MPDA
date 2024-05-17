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
nt = size(XT,2);
% parameter
tow = 10;
d = 4;
sig = 60;
yit = 60;
gam = 1;
kc = 4;
kp = 50;
kt = 50;
yib = 0.6;
sita = 0.8;
bestc = 1;
bestg = 0.25;
accuracy = 0;


%% Model
[Gc,Gp,Gt] = calculation_MPDA(XS,XT,ys,kc,kp,kt,tow);
CDD = zeros(64,1);
for r = 1:100
    [XS_new,XT_new] = reduction(XS,XT,CDD,d,sig,yit,yib,gam,Gc,Gp,Gt);
    
    
    X_train = zscore(XS_new');
    X_test = zscore(XT_new');
    y_train = ys;
    y_test = yt;
    
%% SVM    
    cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg),' -b 1 '];

    model = svmtrain(y_train,X_train,cmd);
    [predict_label,acc,dec_values] = svmpredict(y_test,X_test,model, ' -b 1 ');
    acc_test = length(find(predict_label == y_test))/length(y_test); % accuracy is determined by dividing the number of correctly predicted labels by the total number.
    
    select_label = zeros(nt,1);
    pkx_total = sum(exp(dec_values),2);
    probability_exp = exp(dec_values);
    X_exp = probability_exp ./ pkx_total;
    X_exp =-1 * X_exp .* log(X_exp);
    Loss_ent = sum(X_exp,2);
    Loss_ent = mapminmax(Loss_ent',0, 1);
    t = 1;
    yt_new = zeros(nt,1);
    for lt = 1: nt
        if  Loss_ent(lt) < sita
                                            
            yt_new(lt,1) = predict_label(lt);
                    
            t = t + 1;
         
        end
        
    end
    
    
    [distance] = iteration_owndata_CDD(XS,XT,ys,yt_new);
    CDD = distance;
    
    if accuracy < acc_test 
    
        accuracy = acc_test;
    
    end
    
end

pause(0.2); % delay
clc

disp(['Source domain: Batch ',num2str(1),', Target domain: Batch ',num2str(2),', ','Accuracy = ' num2str(accuracy*100,'%.2f'),'%']);
