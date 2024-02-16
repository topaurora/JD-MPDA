function [CDD] = iteration_CDD(XS,XT,ys,yt_predict)
% Update conditional distribution distance in original space

       d = zeros(128,1); % data feature dimension is 128
       for i = 1:6
           %  calculate the centre of data under label c in source domain
            if size(find(ys == i),1) ~= 0
                XS_i = XS(:,find(ys == i));
                u_sc = sum(XS_i,2)/size(XS_i,2);
            end
           % calculate the centre of data under label c in target domain

            if size(find(yt_predict == i),1) ~= 0 
                XT_i = XT(:,find(yt_predict == i));
                u_tc = sum(XT_i,2)/size(XT_i,2);
            else
                u_tc = zeros(128,1);
            end

            d = d + (u_sc-u_tc)*(u_sc-u_tc)';
       end
       CDD = d;
end

