function [u, u_hat, omega] = MVMD(signal, alpha, tau, K, DC, init, tol) 
% Reference.
%  K. Dragomiretskiy, D. Zosso, Variational Mode Decomposition, IEEE Transactions on Signal Processing, vol. 62, pp. 531-544, 2014. 
%---------- Check for Input Signal              
% Check for getting number of channels from input signal
[x, y] = size(signal);
if x > y
	C = y;% number of channels
    T = x;% length of the Signal
	signal = signal';
else
	C = x;% number of channels
    T = y;% length of the Signal
end
%---------- Preparations
% Sampling Frequency
fs = 1/T;
% Mirroring
f(:,1:T/2) = signal(:,T/2:-1:1);
f(:,T/2+1:3*T/2) = signal;
f(:,3*T/2+1:2*T) = signal(:,T:-1:T/2+1);
% Time Domain 0 to T (of mirrored signal)
T = size(f,2);
t = (1:T)/T;
% frequencies
freqs = t-0.5-1/T;
% Construct and center f_hat
f_hat = fftshift(fft(f,[],2),2);
f_hat_plus = f_hat;
f_hat_plus(:,1:T/2) = 0;
%------------ Initialization
% Maximum number of iterations 
N = 500;
% For future generalizations: individual alpha for each mode
Alpha = alpha*ones(1,K);
% matrix keeping track of every iterant 
u_hat_plus_00 = zeros(length(freqs), C, K);
u_hat_plus = zeros(length(freqs), C, K);
omega_plus = zeros(N, K);
% initialize omegas uniformly
switch init
	case 1
        omega_plus(1,:) = (0.5/K)*((1:K)-1);
    case 2
        omega_plus(1,:) = sort(exp(log(fs) + (log(0.5)-log(fs))*rand(1,K)));
    otherwise
        omega_plus(1,:) = 0;
end
% if DC mode imposed, set its omega to 0
if DC
    omega_plus(1,1) = 0;
end
% start with empty dual variables
lambda_hat = zeros(length(freqs), C, N); 
% other inits
uDiff = tol+eps; % update step
n = 1; % loop counter
sum_uk = zeros(length(freqs), C); % accumulator
%--------------- Algorithm of MVMD
while ( uDiff > tol &&  n < N ) % not converged and below iterations limit	
	% update modes
	for k = 1:K
        % update mode accumulator
        if k > 1
            sum_uk = u_hat_plus(:,:,k-1) + sum_uk - u_hat_plus_00(:,:,k);
        else
            sum_uk = u_hat_plus_00(:,:,K) + sum_uk - u_hat_plus_00(:,:,k);
        end
        % update spectrum of mode through Wiener filter of residuals
		for c = 1:C
			u_hat_plus(:,c,k) = (f_hat_plus(c,:).' - sum_uk(:,c) - lambda_hat(:,c,n)/2)./(1+Alpha(1,k)*(freqs.' - omega_plus(n,k)).^2);
		end
		% update first omega if not held at 0
        if DC || (k > 1)
            % center frequencies
            numerator = freqs(T/2+1:T)*(abs(u_hat_plus(T/2+1:T,:, k)).^2);
            denominator = sum(abs(u_hat_plus(T/2+1:T,:,k)).^2);
            temp1 = sum(numerator);
            temp2 = sum(denominator);
            omega_plus(n+1,k) = temp1/temp2;
        end
	end
	% Dual ascent
    lambda_hat(:,:,n+1) = lambda_hat(:,:,n) + tau*(sum(u_hat_plus,3) - f_hat_plus.');
	% loop counter
	n = n+1;
    u_hat_plus_m1 = u_hat_plus_00;
    u_hat_plus_00 = u_hat_plus;
	% converged yet?
    uDiff = u_hat_plus_00 - u_hat_plus_m1;
    uDiff = 1/T*(uDiff).*conj(uDiff);
	uDiff = eps+abs(sum(uDiff(:)));
end
%------ Post-processing and cleanup
% discard empty space if converged early
N = min(N,n);
omega = omega_plus(1:N,:);
% Signal reconstruction
u_hat = zeros(T, K, C);
for c = 1:C
	u_hat((T/2+1):T,:,c) = squeeze(u_hat_plus((T/2+1):T,c,:));
	u_hat((T/2+1):-1:2,:,c) = squeeze(conj(u_hat_plus((T/2+1):T,c,:)));
	u_hat(1,:,c) = conj(u_hat(end,:,c));
end
u = zeros(K,length(t),C);
for k = 1:K
	for c = 1:C
		u(k,:,c)=real(ifft(ifftshift(u_hat(:,k,c))));
	end
end
% remove mirror part
u = u(:,T/4+1:3*T/4,:);
% recompute spectrum
clear u_hat;
for k = 1:K
	for c = 1:C
		u_hat(:,k,c)=fftshift(fft(u(k,:,c)))';
	end
end
u_hat = permute(u_hat, [2 1 3]);
end