function [XS_new,XT_new] = reduction(XS,XT,CDD,dim,sig,yit,yib,gam,u_intra,u_inter,Gt)
e = 0.1; % the data balance 
% Calculate the center of the source domain and target domain in the original space
us = sum(XS,2)/size(XS,2); 
ut = sum(XT,2)/size(XT,2); 
MMD = (us-ut)*(us-ut)';    % calculate mean distribution discrepancy
[~,m] = size(XS);
[~,n] =size(XT);
XS1 = XS - repmat(us,1,m); % centralisation
XT1 = XT - repmat(ut,1,n); % centralisation

if CDD==0 % In the first cycle, define CDD = 0
    dis = MMD;
else      % dis means the distance of two domains
    dis =   yib* MMD +  (1-yib)* CDD;
end

    
%% Calculate the optimal basis transformation P_
E =  1/m*(XS1*XS1') + e*1/n*(XT1*XT1') - sig * u_intra + yit * u_inter - gam * Gt;%计算源域数据方差、目标域数据方差
A = pinv(dis) * E;         % Calculate matrix A
[U,S] = eig(A);            % Compute eigenvalues and eigenvectors
w = diag(S);         % Form a column of eigenvalues from the diagonal matrix
[~,Index] = sort(w,'descend');      % Sort by ascending order
P_ = real(U(:,Index(1:dim))); %筛选所需的映射矩阵


XS_new = P_' * XS1; % source data in subspace 

XT_new = P_' * XT1; % target data in subspace
end

