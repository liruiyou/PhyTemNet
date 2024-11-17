clc;clear;close all;
global nStrata x_train_data y_train_data  Epochi  MaxEpochi
nStrata = 6;%%layer
pathname=['.\Alldata\RandiData\'];
load([pathname, num2str(nStrata),'LayerHz.mat'])
load([pathname, num2str(nStrata),'LayerRhoh.mat'])
Input= Input(1:250,:);
Output= Output(1:250,:);
Input_theory= Input;
%% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>add noise>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Input=awgn(Input,20,'measured');
Input_noise=Input;
%% ===============================train data==================================
k=rand(1,size(Input,1));
[m,n]=sort(k);
X_training=Input(n(1:end-50),:);
Y_training=Output(n(1:end-50),:);    
X_testing=Input(n(end-50+1:end),:);
Y_testing=Output(n(end-50+1:end),:);
%% =================================================================
x_train_data=X_training';
y_train_data=Y_training';
x_test_data=X_testing';
y_test_data=Y_testing;
[x_train_regular,x_train_maxmin] = mapminmax(x_train_data);
[y_train_regular,y_train_maxmin] = mapminmax(y_train_data);
x_test_regular = mapminmax('apply',x_test_data,x_train_maxmin);
y_train_regular1=y_train_regular';
%%
for i = 1: size(x_train_regular,2) 
    p_train1{i, 1} = (x_train_regular(:,i));	
end	
for i = 1 : size(x_test_regular,2)	
    p_test1{i, 1}  = (x_test_regular(:,i));		
end	
%%
min_batchsize=10;  
layers = [
    sequenceInputLayer(size(x_train_data,1))
    lstmLayer(16,'OutputMode','last')
    selfAttentionLayer(1,2)
    fullyConnectedLayer(size(y_train_data,1))
    % regressionLayer];
    energyConsvLoss("dE")];

options = trainingOptions('adam', ...
    'MaxEpochs',10,...
    'MiniBatchSize',min_batchsize,...
    'InitialLearnRate',0.001,...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.02, ...
     'Plots','training-progress');%%
Epochi = 0; 
MaxEpochi = options.MaxEpochs*(size(X_training,1)/min_batchsize);
[Mdl,Loss] = trainNetwork(p_train1, y_train_regular1, layers, options);
%%
figure
subplot(2, 1, 1)
plot(1 : length(Loss.TrainingRMSE), Loss.TrainingRMSE, 'r-', 'LineWidth', 1)
grid
set(gcf,'color','w')
subplot(2, 1, 2)
plot(1 : length(Loss.TrainingLoss), Loss.TrainingLoss, 'b-', 'LineWidth', 1)
grid
set(gcf,'color','w')
%% 
graph = layerGraph(Mdl.Layers); figure; plot(graph)   
analyzeNetwork(Mdl)  
%% Preserving the training model LSTM-Attention
pathname=['.\TrainModelNet\',num2str(nStrata),'-Layer\'];
if exist(pathname,'dir') == 0
    mkdir(pathname);
end
filename='lstm_attention_net.mat';
save([pathname,filename],'Mdl') ;
%%
y_test_regular=predict(Mdl,p_test1,"MiniBatchSize",min_batchsize);
attention_predict=mapminmax('reverse',y_test_regular',y_train_maxmin);
%%
attention_predict=attention_predict';
errors_nn=sum(abs(attention_predict-y_test_data)./(y_test_data))/length(y_test_data);
figure
color=[111,168,86;128,199,252;112,138,248;184,84,246]/255;
plot(y_test_data,'Color',color(2,:),'LineWidth',1)
hold on
plot(attention_predict,'*','Color',color(1,:))
hold on
titlestr=['lstm-attention',num2str(errors_nn)];
title(titlestr)

rho_Value=y_test_data(50,1:nStrata); h_Valuel=y_test_data(50,nStrata+1:end);
rho_Inv=attention_predict(50,1:nStrata); h_Inv=attention_predict(50,nStrata+1:end);
Hz_Value = zhengyan2(rho_Value,h_Valuel); Hz_Inv = zhengyan2(double(rho_Inv),double(h_Inv));
%%
rou_Theroy=rho_Value';h_Theroy=[h_Valuel inf]';
[d_RouTheroy,d_HTheroy]=SMS_draw_rou(rou_Theroy,h_Theroy);
rou_InvNN=rho_Inv';h_InvNN=[h_Inv inf]';
[d_RouInvNN,d_HInvNN]=SMS_draw_rou(rou_InvNN,h_InvNN);
figure(11);
plot(d_RouTheroy,-d_HTheroy,'k');hold on;
plot(d_RouInvNN,-d_HInvNN,'r--');hold on;
figure(12)
dt = -6:0.1:-3;t = 10.^dt;
loglog(t,Hz_Value,'-ok');hold on;grid on;
loglog(t,abs(Hz_Inv),'-+b');hold on;grid on;
