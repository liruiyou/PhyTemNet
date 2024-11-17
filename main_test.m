%%Deep Neural Network Testing
clc;clear;close all;
nStrata = 3;%%layer
pathname=['.\Alldata\RandiData\'];%Data file location
load([pathname, num2str(nStrata),'LayerHz.mat'])
load([pathname, num2str(nStrata),'LayerRhoh.mat'])
Input= Input(1:250,:);
Output= Output(1:250,:);
Input_theory= Input;
%%  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>add noise>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Input=awgn(Input,20,'measured');
Input_noise=Input;
%% ===============================testing data==================================
k=rand(1,size(Input,1));
[m,n]=sort(k);
X_training=Input(n(1:end-50),:);
Y_training=Output(n(1:end-50),:);    
num=2;%%Selecting Test Data
X_testing=Input(end-num,:);  
Y_testing=Output(end-num,:);  
%% ===========================data normalisation======================================
x_train_data=X_training';
y_train_data=Y_training';
x_test_data=X_testing';
y_test_data=Y_testing;
[x_train_regular,x_train_maxmin] = mapminmax(x_train_data);
[y_train_regular,y_train_maxmin] = mapminmax(y_train_data);
x_test_regular = mapminmax('apply',x_test_data,x_train_maxmin);
y_train_regular1=y_train_regular';
for i = 1: size(x_train_regular,2)    
    p_train1{i, 1} = (x_train_regular(:,i));	
end	
for i = 1 : size(x_test_regular,2)	
    p_test1{i, 1}  = (x_test_regular(:,i));		
end	
%% =======================Loading the training model==========================================
pathname=['.\TrainModelNet\',num2str(nStrata),'-Layer\'];
%% --------VMD-LSTM-Attention-----------------
addpath(genpath(pwd));
load([pathname,'vmd_lstm_attention_net.mat']);
%% predictive modelling
min_batchsize=10;  
y_test_regular=predict(Mdl,p_test1,"MiniBatchSize",min_batchsize);
attention_predict=mapminmax('reverse',y_test_regular',y_train_maxmin);
%% Prediction model error
attention_predict=attention_predict';
errors_nn=sum(abs(attention_predict-y_test_data)./(y_test_data))/length(y_test_data);
figure(1)
color=[111,168,86;128,199,252;112,138,248;184,84,246]/255;
plot(y_test_data,'Color',color(2,:),'LineWidth',1)
hold on
plot(attention_predict,'*','Color',color(1,:))
hold on

rho_Value=y_test_data(1:nStrata); h_Valuel=y_test_data(nStrata+1:end);
rho_Inv=attention_predict(1:nStrata); h_Inv=attention_predict(nStrata+1:end);
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
