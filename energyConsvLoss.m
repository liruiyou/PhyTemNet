classdef energyConsvLoss < nnet.layer.RegressionLayer
    properties

    end
    methods
        function layer = energyConsvLoss(name)
            layer.Name = name;
            layer.Description = 'MAE + Energy Conservation Loss';
        end

        function loss = forwardLoss(layer, Y, T)
            %%=========================================
            global nStrata  x_train_data y_train_data  Epochi  MaxEpochi
            Epochi = Epochi+1;
            Y_predict = 0.5*(Y+1).*repmat((max(y_train_data,[],2)-min(y_train_data,[],2)),1,size(Y,2))+repmat(min(y_train_data,[],2),1,size(Y,2));
            T_true = 0.5*(T+1).*repmat((max(y_train_data,[],2)-min(y_train_data,[],2)),1,size(Y,2))+repmat(min(y_train_data,[],2),1,size(Y,2));
            for i=1:size(Y_predict,2)
                % i
                rho_pred =double(Y_predict(1:nStrata,i)); h_pred=double(Y_predict(nStrata+1:end,i));
                rho_true =double(T_true(1:nStrata,i)); h_true=double(T_true(nStrata+1:end,i));
                Hz_pred(:,i) = zhengyan2(rho_pred, h_pred);%%
                Hz_true(:,i) = zhengyan2(rho_true, h_true);
                % Hz_pred(:,i) = ELMTest(rho_pred, h_pred,nStrata);%%
                % Hz_true(:,i) = ELMTest(rho_true, h_true,nStrata);
                Hz_true = single(Hz_true);
            end
            % figure(1)
            %     dt = -6:0.1:-3;t = 10.^dt;
            %     loglog(t,Hz_true(:,1:2:10),'-ok');hold on;grid on;
            %     loglog(t,extractdata(Hz_pred(:,1:2:10)),'-+b');hold on;grid on;
            Hz_Pred = 2*(Hz_pred-repmat(min(x_train_data,[],2),1,size(Hz_pred,2)))./(repmat(max(x_train_data,[],2),1,size(Hz_pred,2))-repmat(min(x_train_data,[],2),1,size(Hz_pred,2)))-1;
            Hz_True = 2*(Hz_true-repmat(min(x_train_data,[],2),1,size(Hz_true,2)))./(repmat(max(x_train_data,[],2),1,size(Hz_pred,2))-repmat(min(x_train_data,[],2),1,size(Hz_pred,2)))-1;
            % MAEloss = mean(abs( Hz_Pred(:)-Hz_True(:)));
            MAEloss = mean(abs( Hz_Pred(:)-Hz_True(:)))+2*(1-Epochi/(2*MaxEpochi)).^0.5*mean(mean(abs(diff(extractdata(Y)))));%
            loss = MAEloss;
        end
    end
end