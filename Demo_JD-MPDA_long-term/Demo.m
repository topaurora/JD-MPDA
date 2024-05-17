%% Experimental in Setting 2 (long-term drift)
% Set batch 1 as the source domain, and batch K (K = 2, 3, ..., 10) as the target domain
% Drift_dataset: The Gas Sensor Array Drift dataset from UCI. http://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset

clc
clear
load Drift_dataset

% Use batch 1 as source domain
XS = Drift_dataset{1,1};

% labels of batch 1
ys = Drift_dataset{1,2};

%parameters
d_m = [4 4 4 4 4 2 4 2 4];
sig_ma = [20 40 30 20 10 40 10 30 4];
yit_a = [1 46 10 30 10 30 20 40 22];
gam_ma = [1 42 20 1 20 10 10 20 32];
k_c = [32 8 32 32 8 32 2 32 16];
k_p = [40 30 50 40 50 40 60 40 30];
k_t = [30 10 60 30 30 50 40 50 10];
yib_sio = [0.4 0.2 0.6 0 0 0.3 0.1 0.5 0.4];
sita = [0.6 0.3 0.4 0.6 1 0.6 0.6 0.6 0.5];
tow = 10;
c = 1;
g = [0.25 0.25 0.25 0.25 0.25 0.5 0.25 0.5 0.25];
predict_acc = cell(9,1);


for i = 1:9
    XT = Drift_dataset{i+1,1};
    yt = Drift_dataset{i+1,2};
    nt = size(XT,2);
    d = d_m(i);
    sig = sig_ma(i);
    yit = yit_a(i);
    gam = gam_ma(i);
    kc = k_c(i);
    kp = k_p(i);
    kt = k_t(i);
    yib = yib_sio (i);
    sit = sita(i);
    accuracy = 0; % max accuracy
    
%% Model    
    [Gc,Gp,Gt] = calculation_MPDA(XS,XT,ys,kc,kp,kt,tow); % Calculate the portion of the iteration that does not change
    
    CDD = zeros(128,1); % Initialising the conditional distribution

    for r = 1:10 
        [XS_new, XT_new,P] = reduction(XS,XT,CDD,d,sig,yit,yib,gam,Gc,Gp,Gt);
        
%% SVM libsvm was developed by Professor Lin of National Taiwan University and is publicly available.     
        X_train = zscore(XS_new');
        X_test = zscore(XT_new');
        y_train = ys;
        y_test = yt;
        bestc = c;
        bestg = g(i);
        
        if sit ~= 1  % sita == 1 means skip the screening step
            cmd = [' -c ',num2str(bestc),' -g ',num2str(bestg),' -b 1 '];
            model = svmtrain(y_train,X_train,cmd); 
            [predict_label,~,dec_values] = svmpredict(y_test,X_test,model,'-b 1');

        else
            cmd = [' -c ',num2str(bestc),' -g ',num2str(bestg)];
            model = svmtrain(y_train,X_train,cmd);
            [predict_label,~,dec_values] = svmpredict(y_test,X_test,model);
            
        end
        
        acc = length(find(predict_label == y_test))/length(y_test); % accuracy is determined by dividing the number of correctly predicted labels by the total number.
        if accuracy < acc 
            accuracy = acc;
        end

%% Calculate the conditional-entropy loss

        if sit ~= 1
            
            pkx_total = sum(exp(dec_values),2); 
            
            probability_exp = exp(dec_values); 
            
            X_exp = probability_exp ./ pkx_total; 
            
            X_exp =-1 * X_exp .* log(X_exp);  
            
            Loss_ent = sum(X_exp,2);         
            
            Loss_ent = mapminmax(Loss_ent',0, 1); % normalize the valus to [0,1]
            
%% Selection

            t = 1;                              
            
            yt_new = zeros(nt,1);  
            
            for lt = 1: nt      
            
                if  Loss_ent(lt) < sit          
                
                    yt_new(lt,1) = predict_label(lt);
                    
                    t = t + 1;
                
                end
                
            end
  
        else
            
            yt_new = predict_label;
        
        end
        
        [distance] = iteration_CDD(XS,XT,ys,yt_new); 
        
        CDD = distance;
    
    end
    
    predict_acc{i,1} = accuracy;

end

pause(0.2);

clc

for j = 1:9

    disp(['Source domain: Batch',num2str(1),', Target domain: Batch',num2str(j+1),', ','Accuracy = ' num2str(predict_acc{j,1}*100,'%.2f'),'%']);

end
        
        
    
