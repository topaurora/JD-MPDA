function [CDD] = iteration_CQU_CDD(XS,XT,ys,yt_predict)


       d = zeros(6,1); % data feature dimension of CQU dataset is 6
       for i = 1:6
            if size(find(ys == i),1) ~= 0
                XS_i = XS(:,find(ys == i));
                u_sc = sum(XS_i,2)/size(XS_i,2);
            end
            
            if size(find(yt_predict == i),1) ~= 0
                XT_i = XT(:,find(yt_predict == i));
                u_tc = sum(XT_i,2)/size(XT_i,2);
            else
                u_tc = zeros(6,1);
            end

            d = d + (u_sc-u_tc)*(u_sc-u_tc)';
       end
       CDD = d;
end
