function [Gc,Gp,Gt] = calculation(XS,XT,ys,kcc,kp,kt,tow)
% This function aims to calculate Gc, Gp and Gt which have no change in cycle. 
% Gc, Gp and Gt use the same method of Laplace matrix composition.
%%   Data standardisation for easy comparison
XS = zscore(XS'); 
XS = XS';
XT = zscore(XT');
XT = XT';
[~,m] = size(XS); % m means the total number of source data

%% Intraclass compactness
distense = zeros(m,m);
table = zeros(m,m);
Wc = zeros(m,m);
Dc = zeros(m,m);
I_all = zeros(m,m);
for i = 1:m  % Iterate over all data
    XS_i = XS(:,i); % the ith data 
    ys_i = ys(i);   
    k =1;
    for j = 1:m   % Iterate over all the data again
        if i ~= j   % Selection of data other than the ith
            if ys(i,:) == ys(j,:)  % if the ith and jth labels are the same
                distense(i,k) = norm(XS_i-XS(:,j),2); % calculate the distance between the ith data and the jth data
                table (i,k) = j; % retain the corresponding location information
                k = k+1;   %  the total number of distance for each data is n-1,n means the number of data under label c.
            end
        end
    end
    t = find(ys == ys_i); % location of data under label c
    [n,~] = size(t); % the number of data under label c
    [~,I] = sort(distense(i,1:n-1)); % sort distances from smallest to largest

    if n-1 <= kcc % If the neighbourhood range is greater than n-1, n-1 is used as the upper neighbourhood value
        kc = n-1;
    else
        kc = kcc;
    end
    for lp = 1:kc % filter the neighborhood
        u = I(lp);  
% At this point, the position corresponding to u is in the order from smallest to largest in the distinence. 
%For example, when lp=1, now u=8.It means that distence（i,8）is the smallest
        v = table(i,u); % find the original location
        Wc(i,v) = 1;   % assign values according to equation (6)
    end
end

for llp = 1:m
    for lop = 1:m
        Dc(llp,llp) = Dc(llp,llp) + Wc(llp,lop); % calculate degree matrix
    end
end
Gc = 2 * XS * (Dc - Wc) * XS'; % the same as equation (5)


%% interclass separability
Wp = zeros(m,m);
Dp = zeros(m,m);
dist = zeros(m,m);
tabe = zeros(m,m);
I_inn = zeros(m,m);
for i = 1:m
    XS_i = XS(:,i); 
    ys_i = ys(i,:); 
    k = 1;
    for j = 1:m
        if i ~= j
            if ys(i,:) ~= ys(j,:)   % find data with different labels than the ith data
                dist(i,k) =  norm(XS_i-XS(:,j),2); 
                tabe(i,k) = j; % record the original position
                k = k+1;
            end
        end
    end
    tp = find(ys ~= ys_i); % number of statistics not belonging to category i
    [nt,~] = size(tp);
    [~,Ip] = sort(dist(i,1:nt)); 
    
    for dt = 1:kp  % filter the neighborhood
        dtt = Ip(dt);
        v = tabe(i,dtt);
        Wp(i,v) = 1;  % assign values according to equation (8)
    end
end
for llp = 1:m
    for lop = 1:m
        Dp(llp,llp) = Dp(llp,llp) + Wp(llp,lop); % calculate degree matrix
    end
end
Gp = 2 * XS * (Dp - Wp) * XS'; % the same as equation (7) 



%% manifold structure preserving Unsupervised
[~,mt] = size(XT);
dis = zeros(mt,mt-1);
loc = zeros(mt,mt-1);
Wt = zeros(mt,mt);
Dt = zeros(mt,mt);

for i = 1:mt %Iterate over all data
    XT_i = XT(:,i); % the ith data
    k = 1;
    for j = 1:mt
        if i ~= j 
            dis(i,k) = norm(XT_i - XT(:,j),2);
            loc(i,k) = j; % record the original position
            k = k+1;
        end
    end
    [~,It] = sort(dis(i,:)); 
    for lp = 1:kt
        llp = It(lp);
        power = abs(dis(i,llp))/tow; % hot kernel 'tow' is 10
        wave = exp(-1 * power); % 
        o = loc(i,llp); 
        Wt(i,o) = wave; % assign values according to equation (10)
    end
end
for i = 1:mt
    for j = 1:mt
        Dt(i,i) = Dt(i,i) + Wt(i,j);
    end
end
L = Dt - Wt;
Gt = XT * L * XT'; % same as equation (9)
end
