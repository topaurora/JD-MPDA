function [CDD] = iteration_owndata_CDD(XS,XT,ys,yt_predict)
       d = zeros(64,1); % data feature dimension is 64
       for i = 1:5

            if size(find(yt_predict == i),1) ~= 0
                XT_i = XT(:,find(yt_predict == i));
                u_tc = sum(XT_i,2)/size(XT_i,2);
            else
                u_tc = zeros(64,1);
            end
            if size(find(ys == i),1) ~= 0
                XS_i = XS(:,find(ys == i));
                u_sc = sum(XS_i,2)/size(XS_i,2);
            end
            d = d + (u_sc-u_tc)*(u_sc-u_tc)';
       end
       CDD = d;
end

