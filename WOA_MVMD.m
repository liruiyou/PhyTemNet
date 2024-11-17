% The objective function is the local envelope entropy minima % The optimisation variables are the penalty coefficient and the number of decomposition layers (¦Á and K)
clc;clear;close all;warning off
nStrata = 5;%%layer
pathname=['.\Alldata\RandiData\'];
load([pathname, num2str(nStrata),'LayerHz.mat'])
load([pathname, num2str(nStrata),'LayerRhoh.mat'])
Input= Input(1:250,:);
Output= Output(1:250,:);
Input_theory= Input;
%% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>add boise>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Input=awgn(Input,20,'measured');
Input_noise=Input;
%%=============================WOA-MVMD denoising==================================
% for num=1:size(Input,1)
fs=1;%
Ts=1/fs;%
X = (Input(:,1:end))';
L=size(X,1);%
t=(0:L-1)*Ts;
STA=0; 
tau = 0;              
DC = 0;              
init = 1;              
tol = 1e-7;  
%% Optimisation of the best MVMD parameters ¦Á and K using the whale optimisation algorithm
SearchAgents_no=5; 
Max_iteration=30; 
dim=2; 
lb=[2,200]; 
ub=[10,2500]; 
Leader_pos=zeros(1,dim);
Leader_score=inf;
Positions=initialization(SearchAgents_no,dim,ub,lb);
Convergence_curve=zeros(1,Max_iteration);
t=1;% 
while t<Max_iteration+1
    for i=1:size(Positions,1)
        Flag4ub=Positions(i,:)>ub;
        Flag4lb=Positions(i,:)<lb;
        Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        Positions(i,:)=round( Positions(i,:));
        % [u, ~, ~] = VMD(X,  Positions(i,2), tau,  Positions(i,1), DC, init, tol);
        [u, ~, ~] = MVMD(X, Positions(i,2), tau, Positions(i,1), DC, init, tol);
        u=reshape(mean(u,2),size(u,1),31);

        for ii=1:Positions(i,1)
            bao=hilbert(u(ii,:));
            bao=abs(bao);
            p=bao./sum(bao);
            e1(ii,:)=-sum(p.*log10(p));
        end
        fitness=min(e1);
        if fitness<Leader_score %
            Leader_score=fitness; % 
            Leader_pos=Positions(i,:);
        end
    end
    a=2-t*((2)/Max_iteration); 
    a2=-1+t*((-1)/Max_iteration);
    for i=1:size(Positions,1)
        r1=rand(); % r1 is a random number in [0,1]
        r2=rand(); % r2 is a random number in [0,1]
        A=2*a*r1-a;  % Eq. (2.3) in the paper
        C=2*r2;      % Eq. (2.4) in the paper
        b=1;               %  parameters in Eq. (2.5)
        l=(a2-1)*rand+1;   %  parameters in Eq. (2.5)
        p = rand();        % p in Eq. (2.6)
        for j=1:size(Positions,2)
            if p<0.5
                if abs(A)>=1
                    rand_leader_index = floor(SearchAgents_no*rand()+1);
                    X_rand = Positions(rand_leader_index, :);
                    D_X_rand=abs(C*X_rand(j)-Positions(i,j));
                    Positions(i,j)=X_rand(j)-A*D_X_rand;
                elseif abs(A)<1
                    D_Leader=abs(C*Leader_pos(j)-Positions(i,j));
                    Positions(i,j)=Leader_pos(j)-A*D_Leader;
                end
            elseif p>=0.5
                distance2Leader=abs(Leader_pos(j)-Positions(i,j));
                Positions(i,j)=distance2Leader*exp(b.*l).*cos(l.*2*pi)+Leader_pos(j);
            end
        end
    end

    Convergence_curve(t)=Leader_score;
    t=t+1;
end
besta=Leader_pos(1,1);
bestK=Leader_pos(1,2);
bestGWOaccuarcy=Leader_score;
%% 
[u, u_hat, omega] = MVMD(X,  bestK, tau,  besta, DC, init, tol);
input_MVMD=u;
Input_MVMD = reshape(abs(input_MVMD(1,:,:)),size(input_MVMD,2),size(input_MVMD,3)); 

num=40;
figure(11)
dt = -6:0.1:-3;t = 10.^dt;
loglog(t,Input_theory(num,:),'-ok');hold on;grid on;
loglog(t,abs(Input_noise(num,:)),'-+b');hold on;grid on;
loglog(t,abs(Input_MVMD(num,:)),'-+r');hold on;grid on;
figure(12)
dt = -6:0.1:-3;t = 10.^dt;
loglog(t,Input_theory(num,:),'-ok');hold on;grid on;
Variance=Input_noise(num,:)-Input_theory(1,:);
errorbar(t,Input_theory(num,:),Variance,'-+b');
loglog(t,abs(Input_MVMD(num,:)),'-+r');hold on;grid on;

figure(1)
plot(Convergence_curve, 'r-', 'LineWidth',1.0)
grid on
num = 1; 
X = X(:,num);
t = 1 : length(X);
figure(2)
u_mvmd=u;
u= reshape(u_mvmd(:,num,:),size(u_mvmd,1),size(u_mvmd,3));
imfn=u;
n=size(imfn,1); 
subplot(n+1,1,1);  
plot(t,X); 
for n1=1:n
    subplot(n+1,1,n1+1);
    plot(t,u(n1,:));
    ylabel(['IMF' int2str(n1)]);
end

[m,n]=size(u);
m=m+1;
t = 1 : length(X);
fs = 1;
figure(3)
for i=1:m-1
    subplot(m,1,i)
    plot(t,u(i,:),'k-','linewidth',1)
    ylabel(['IMF',num2str(i)]);
end
res = X'-sum(u,1);
subplot(m,1,m)
plot(t,res,'k-','linewidth',1)
ylabel('Res');xlabel('t/s');

%
figure(4)
for i=1:m-1
    subplot(m,1,i)
    [cc,y_f]=plot_fft(u(i,:),fs,1);
    plot(y_f,cc,'k','LineWIdth',1);
    ylabel(['FFT of IMF',num2str(i)]);
end
subplot(m,1,m)
[cc,y_f]=plot_fft(res,fs,1);
plot(y_f,cc,'k','LineWIdth',1);
ylabel('FFT of Res');xlabel('f/Hz');


















