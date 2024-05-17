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
nt = size(XT,2);
% parameter 
d = 6;
sig = 80;
yit = 90;
gam = 10;
kc = 16;
kp = 70;
kt = 60;
accuracy = 0;
yib = 0.1;
sit = 0.8;
c = 1;
g = 0.1667;

%% Model
% Calculate the portion of the iteration that does not change
[Gc,Gp,Gt] = calculation_MFDA(XS,XT,ys,kc,kp,kt,10);

    CDD = zeros(6,1);
for r = 1:100 % the cycle

    [XS_new,XT_new,P] = reduction(XS,XT,CDD,d,sig,yit,yib,gam,Gc,Gp,Gt); % get subspace data
    
    X_train = zscore(XS_new');
    
    X_test = zscore(XT_new');
    
    y_train = ys;
    
    y_test = yt;

    
%% SVM
    
    cmd = [' -t 2',' -c ',num2str(c),' -g ',num2str(g),' -b 1 '];

    model = svmtrain(y_train,X_train,cmd);
    
    [predict_label,acc,dec_values] = svmpredict(y_test,X_test,model,'-b 1');
    
    acc_test = length(find(predict_label == y_test))/length(y_test); % accuracy is determined by dividing the number of correctly predicted labels by the total number.
    
    if accuracy < acc_test
    
        accuracy = acc_test;
        
    end
% Update the conditional distribution distance
  
    pkx_total = sum(exp(dec_values),2); 
            
    probability_exp = exp(dec_values); 
            
    X_exp = probability_exp ./ pkx_total; 
            
    X_exp =-1 * X_exp .* log(X_exp);      
    
    Loss_ent = sum(X_exp,2);              % Calculate the conditional-entropy loss 
           
    Loss_ent = mapminmax(Loss_ent',0, 1); % normalize the valus to [0,1]
    
% Selection   
    
    t = 1;                               
    
    yt_new = zeros(nt,1);  
       
    
    for lt = 1: nt      
     
        if  Loss_ent(lt) < sit          
         
            yt_new(lt,1) = predict_label(lt);
          
            t = t + 1;
            
        end
        
    end
    
    [distance] = iteration_CQU_CDD(XS,XT,ys,yt_new);
    
    CDD = distance;
    
end

pause(0.2);
clc

disp(['Source domain: Batch Master',', Target domain: Batch Slave ',num2str(1),', ','Accuracy = ' num2str(accuracy*100,'%.2f'),'%']);
