%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%    Massive-MIMO Uplink and Downlink Beamforming Simulation
%
%	Author(s): Rahman Doost-Mohamamdy: doost@rice.edu
%
%---------------------------------------------------------------------
% Original code copyright Mango Communications, Inc.
% Distributed under the WARP License http://warpproject.org/license
% Copyright (c) 2018-2019, Rice University
% RENEW OPEN SOURCE LICENSE: http://renew-wireless.org/license
% ---------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all;


% Waveform params
N_OFDM_SYMS             = 24;           % Number of OFDM symbols
MOD_ORDER               = 16;           % Modulation order (2/4/16/64 = BSPK/QPSK/16-QAM/64-QAM)
TX_SCALE                = 1.0;          % Scale for Tx waveform ([0:1])

% OFDM params
SC_IND_PILOTS           = [8 22 44 58];                           % Pilot subcarrier indices
SC_IND_DATA             = [2:7 9:21 23:27 39:43 45:57 59:64];     % Data subcarrier indices
N_SC                    = 64;                                     % Number of subcarriers
N_DATA_SYMS             = N_OFDM_SYMS * length(SC_IND_DATA);      % Number of data symbols (one per data-bearing subcarrier per OFDM symbol)

SAMP_FREQ               = 20e6;

% Massive-MIMO params
N_UE                    = 1;
N_BS_ANT                = 16;               % N_BS_ANT >> N_UE
N_0                     = 1e-2;
H_var                   = 0.1;

u_num = 1;
tti = 28;
H_r = hdf5read('./16_4_16_2.hdf5', '/H_r');
H_i = hdf5read('./16_4_16_2.hdf5', '/H_i');
H = reshape(H_r(tti,:,u_num)+ H_i (tti,:,u_num)* 1i,[16,1]);



% LTS for CFO and channel estimation
lts_f = [0 1 -1 -1 1 1 -1 1 -1 1 -1 -1 -1 -1 -1 1 1 -1 -1 1 -1 1 -1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 -1 -1 1 1 -1 1 -1 1 1 1 1 1 1 -1 -1 1 1 -1 1 -1 1 1 1 1];
lts_t = ifft(lts_f, 64);

pilot_in_mat = zeros(N_UE, N_SC, N_UE);
for i=1:N_UE
    pilot_in_mat (i, :, i) = lts_f;
end

lts_f_mat = zeros(N_BS_ANT, N_SC, N_UE);
for i = 1:N_UE
    lts_f_mat(:, :, i) = repmat(lts_f, N_BS_ANT, 1);
end

%% Modulation & Demodulation functions
% Functions for data -> complex symbol mapping (like qammod, avoids comm toolbox requirement)
% These anonymous functions implement the modulation mapping from IEEE 802.11-2012 Section 18.3.5.8

modvec_bpsk   =  (1/sqrt(2))  .* [-1 1];
modvec_16qam  =  (1/sqrt(10)) .* [-3 -1 +3 +1];
modvec_64qam  =  (1/sqrt(43)) .* [-7 -5 -1 -3 +7 +5 +1 +3];

mod_fcn_bpsk  = @(x) complex(modvec_bpsk(1+x),0);
mod_fcn_qpsk  = @(x) complex(modvec_bpsk(1+bitshift(x, -1)), modvec_bpsk(1+mod(x, 2)));
mod_fcn_16qam = @(x) complex(modvec_16qam(1+bitshift(x, -2)), modvec_16qam(1+mod(x,4)));
mod_fcn_64qam = @(x) complex(modvec_64qam(1+bitshift(x, -3)), modvec_64qam(1+mod(x,8)));

demod_fcn_bpsk = @(x) double(real(x)>0);
demod_fcn_qpsk = @(x) double(2*(real(x)>0) + 1*(imag(x)>0));
demod_fcn_16qam = @(x) (8*(real(x)>0)) + (4*(abs(real(x))<0.6325)) + (2*(imag(x)>0)) + (1*(abs(imag(x))<0.6325));
demod_fcn_64qam = @(x) (32*(real(x)>0)) + (16*(abs(real(x))<0.6172)) + (8*((abs(real(x))<(0.9258))&&((abs(real(x))>(0.3086))))) + (4*(imag(x)>0)) + (2*(abs(imag(x))<0.6172)) + (1*((abs(imag(x))<(0.9258))&&((abs(imag(x))>(0.3086)))));

%% Uplink

% Generate a payload of random integers
tx_ul_data = randi(MOD_ORDER, N_UE, N_DATA_SYMS) - 1;

% Map the data values on to complex symbols
switch MOD_ORDER
    case 2         % BPSK
        tx_ul_syms = arrayfun(mod_fcn_bpsk, tx_ul_data);
    case 4         % QPSK
        tx_ul_syms = arrayfun(mod_fcn_qpsk, tx_ul_data);
    case 16        % 16-QAM
        tx_ul_syms = arrayfun(mod_fcn_16qam, tx_ul_data);
    case 64        % 64-QAM
        tx_ul_syms = arrayfun(mod_fcn_64qam, tx_ul_data);
    otherwise
        fprintf('Invalid MOD_ORDER (%d)!  Must be in [2, 4, 16, 64]\n', MOD_ORDER);
        return;
end

% Reshape the symbol vector to a matrix with one column per OFDM symbol
tx_ul_syms_mat = reshape(tx_ul_syms, N_UE, length(SC_IND_DATA), N_OFDM_SYMS);

% Define the pilot tone values as BPSK symbols
pt_pilots = [1 1 -1 1].';

% Repeat the pilots across all OFDM symbols
pt_pilots_mat = zeros(N_UE, 4, N_OFDM_SYMS);
for i=1:N_UE
    pt_pilots_mat(i,:,:) = repmat(pt_pilots, 1, N_OFDM_SYMS);
end
tic
% Construct the IFFT input matrix
data_in_mat = zeros(N_UE, N_SC, N_OFDM_SYMS);

% Insert the data and pilot values; other subcarriers will remain at 0
data_in_mat(:, SC_IND_DATA, :) = tx_ul_syms_mat;
data_in_mat(:, SC_IND_PILOTS, :) = pt_pilots_mat;

tx_mat_f = cat(3, pilot_in_mat, data_in_mat);
tx_payload_vec = reshape(tx_mat_f, N_UE, numel(tx_mat_f(1,:,:)));

% Rayleight + AWGN:

rng('shuffle');

Z_mat = sqrt(N_0/2) * ( randn(N_BS_ANT,length(tx_payload_vec) ) + 1i*randn(N_BS_ANT,length(tx_payload_vec) ) );     % UL noise matrix
% H = sqrt(H_var/2) .* ( randn(N_BS_ANT, N_UE) + 1i*randn(N_BS_ANT, N_UE) );                                  % Spatial Channel Matrix

rx_payload_vec = H * tx_payload_vec + Z_mat;
rx_mat_f = reshape(rx_payload_vec, N_BS_ANT, N_SC, N_UE + N_OFDM_SYMS);

csi_mat = rx_mat_f(:, :, 1:N_UE) .* lts_f_mat;
fft_out_mat = rx_mat_f(:, :, N_UE+1:end);

demult_mat = zeros(N_UE, N_SC, N_OFDM_SYMS);
sc_csi_mat = zeros(N_BS_ANT, N_UE);
for j=1:N_SC
    sc_csi_mat = squeeze(csi_mat(:, j, :));
    zf_mat = pinv(sc_csi_mat);
    demult_mat(:, j, :) = zf_mat * squeeze(fft_out_mat(:, j, :));
end

payload_syms_mat = demult_mat(:, SC_IND_DATA, :);
payload_syms_mat = reshape(payload_syms_mat, N_UE, numel(payload_syms_mat(1,:,:)));
toc

tx_ul_syms_vecs = reshape(tx_ul_syms_mat, N_UE, numel(tx_ul_syms_mat(1, :, :)));
ul_evm_mat = abs(payload_syms_mat - tx_ul_syms_vecs).^2;
ul_aevms = mean(ul_evm_mat, 2);
ul_snrs = 10*log10(1 ./ ul_aevms);

for n_ue = 1:N_UE
    if ul_snrs(n_ue)>0
        ul_snr_d(n_ue) = ul_snrs(n_ue);
    else 
        ul_snr_d(n_ue) = 0;
    end
end

ul_se = zeros(N_UE);
for n_ue = 1:N_UE
    ul_se(n_ue) = log2(MOD_ORDER) * log2(1+10.^(ul_snr_d(n_ue)/10));
end
ul_se_total = sum(ul_se);

% se = log2(MOD_ORDER) * log2((1 + 10.^(ul_snrs/10)));

%% Plots:
cf = 0;

% UL
cf = cf + 1;
figure(cf); clf;
subplot(2,2,1)
plot(payload_syms_mat(1, :),'ro','MarkerSize',1);
axis square; axis(1.5*[-1 1 -1 1]);
grid on;
hold on;

plot(tx_ul_syms(1, :),'bo');
title('Uplink Tx and Rx Constellations')
legend('Rx','Tx');


% subplot(2,2,2)
% plot(payload_syms_mat(2, :),'ro','MarkerSize',1);
% axis square; axis(1.5*[-1 1 -1 1]);
% grid on;
% hold on;
% 
% plot(tx_ul_syms(2, :),'bo');
% legend('Rx','Tx');
% 
% 
% subplot(2,2,3)
% plot(payload_syms_mat(3, :),'ro','MarkerSize',1);
% axis square; axis(1.5*[-1 1 -1 1]);
% grid on;
% hold on;
% 
% plot(tx_ul_syms(3, :),'bo');
% legend('Rx','Tx');
% 
% 
% subplot(2,2,4)
% plot(payload_syms_mat(4, :),'ro','MarkerSize',1);
% axis square; axis(1.5*[-1 1 -1 1]);
% grid on;
% hold on;
% 
% plot(tx_ul_syms(4, :),'bo');
% legend('Rx','Tx');

% DL
% cf = cf + 1;
% figure(cf); clf;
% subplot(2,2,1)
% plot(payload_dl_syms_mat(1, :),'ro','MarkerSize',1);
% axis square; axis(1.5*[-1 1 -1 1]);
% grid on;
% hold on;
% 
% plot(tx_dl_syms(1, :),'bo');
% title('Downlink Tx and Rx Constellations')
% legend('Rx','Tx');
% 
% 
% subplot(2,2,2)
% plot(payload_dl_syms_mat(2, :),'ro','MarkerSize',1);
% axis square; axis(1.5*[-1 1 -1 1]);
% grid on;
% hold on;
% 
% plot(tx_dl_syms(2, :),'bo');
% legend('Rx','Tx');
% 
% 
% subplot(2,2,3)
% plot(payload_dl_syms_mat(3, :),'ro','MarkerSize',1);
% axis square; axis(1.5*[-1 1 -1 1]);
% grid on;
% hold on;
% 
% plot(tx_dl_syms(3, :),'bo');
% legend('Rx','Tx');
% 
% 
% subplot(2,2,4)
% plot(payload_dl_syms_mat(4, :),'ro','MarkerSize',1);
% axis square; axis(1.5*[-1 1 -1 1]);
% grid on;
% hold on;
% 
% plot(tx_dl_syms(4, :),'bo');
% legend('Rx','Tx');

% EVM & SNR UL
cf = cf + 1;
figure(cf); clf;


subplot(2,2,1)
plot(100*ul_evm_mat(1,:),'o','MarkerSize',1)
axis tight
hold on
plot([1 length(ul_evm_mat(1,:))], 100*[ul_aevms(1,:), ul_aevms(1,:)],'r','LineWidth',2)
title('Downlink Rx Stats')
myAxis = axis;
h = text(round(.05*length(ul_evm_mat(1,:))), 100*ul_aevms(1,:)+ .1*(myAxis(4)-myAxis(3)), sprintf('Effective SNR: %.1f dB', ul_snrs(1,:)));
set(h,'Color',[1 0 0])
set(h,'FontWeight','bold')
set(h,'FontSize',10)
set(h,'EdgeColor',[1 0 0])
set(h,'BackgroundColor',[1 1 1])
hold off
xlabel('Data Symbol Index')
ylabel('EVM (%)');
legend('Per-Symbol EVM','Average EVM','Location','NorthWest');
title('EVM vs. Data Symbol Index')
grid on


% subplot(2,2,2)
% plot(100*ul_evm_mat(2,:),'o','MarkerSize',1)
% axis tight
% hold on
% plot([1 length(ul_evm_mat(2,:))], 100*[ul_aevms(2,:), ul_aevms(2,:)],'r','LineWidth',2)
% myAxis = axis;
% h = text(round(.05*length(ul_evm_mat(2,:))), 100*ul_aevms(2,:)+ .1*(myAxis(4)-myAxis(3)), sprintf('Effective SNR: %.1f dB', ul_snrs(2,:)));
% set(h,'Color',[1 0 0])
% set(h,'FontWeight','bold')
% set(h,'FontSize',10)
% set(h,'EdgeColor',[1 0 0])
% set(h,'BackgroundColor',[1 1 1])
% hold off
% xlabel('Data Symbol Index')
% ylabel('EVM (%)');
% legend('Per-Symbol EVM','Average EVM','Location','NorthWest');
% title('EVM vs. Data Symbol Index')
% grid on
% 
% subplot(2,2,3)
% plot(100*ul_evm_mat(3,:),'o','MarkerSize',1)
% axis tight
% hold on
% plot([1 length(ul_evm_mat(3,:))], 100*[ul_aevms(3,:), ul_aevms(3,:)],'r','LineWidth',2)
% myAxis = axis;
% h = text(round(.05*length(ul_evm_mat(3,:))), 100*ul_aevms(3,:)+ .1*(myAxis(4)-myAxis(3)), sprintf('Effective SNR: %.1f dB', ul_snrs(3,:)));
% set(h,'Color',[1 0 0])
% set(h,'FontWeight','bold')
% set(h,'FontSize',10)
% set(h,'EdgeColor',[1 0 0])
% set(h,'BackgroundColor',[1 1 1])
% hold off
% xlabel('Data Symbol Index')
% ylabel('EVM (%)');
% legend('Per-Symbol EVM','Average EVM','Location','NorthWest');
% title('EVM vs. Data Symbol Index')
% grid on
% 
% subplot(2,2,4)
% plot(100*ul_evm_mat(4,:),'o','MarkerSize',1)
% axis tight
% hold on
% plot([1 length(ul_evm_mat(4,:))], 100*[ul_aevms(4,:), ul_aevms(4,:)],'r','LineWidth',2)
% myAxis = axis;
% h = text(round(.05*length(ul_evm_mat(4,:))), 100*ul_aevms(4,:)+ .1*(myAxis(4)-myAxis(3)), sprintf('Effective SNR: %.1f dB', ul_snrs(4,:)));
% set(h,'Color',[1 0 0])
% set(h,'FontWeight','bold')
% set(h,'FontSize',10)
% set(h,'EdgeColor',[1 0 0])
% set(h,'BackgroundColor',[1 1 1])
% hold off
% xlabel('Data Symbol Index')
% ylabel('EVM (%)');
% legend('Per-Symbol EVM','Average EVM','Location','NorthWest');
% title('EVM vs. Data Symbol Index')
% grid on

% EVM & SNR DL
% cf = cf + 1;
% figure(cf); clf;
% 
% 
% subplot(2,2,1)
% plot(100*dl_evm_mat(1,:),'o','MarkerSize',1)
% axis tight
% hold on
% plot([1 length(dl_evm_mat(1,:))], 100*[dl_aevms(1,:), dl_aevms(1,:)],'r','LineWidth',2)
% title('Downlink Rx Stats')
% myAxis = axis;
% h = text(round(.05*length(dl_evm_mat(1,:))), 100*dl_aevms(1,:)+ .1*(myAxis(4)-myAxis(3)), sprintf('Effective SNR: %.1f dB', dl_snrs(1,:)));
% set(h,'Color',[1 0 0])
% set(h,'FontWeight','bold')
% set(h,'FontSize',10)
% set(h,'EdgeColor',[1 0 0])
% set(h,'BackgroundColor',[1 1 1])
% hold off
% xlabel('Data Symbol Index')
% ylabel('EVM (%)');
% legend('Per-Symbol EVM','Average EVM','Location','NorthWest');
% title('EVM vs. Data Symbol Index')
% grid on
% 
% 
% subplot(2,2,2)
% plot(100*dl_evm_mat(2,:),'o','MarkerSize',1)
% axis tight
% hold on
% plot([1 length(dl_evm_mat(2,:))], 100*[dl_aevms(2,:), dl_aevms(2,:)],'r','LineWidth',2)
% myAxis = axis;
% h = text(round(.05*length(dl_evm_mat(2,:))), 100*dl_aevms(2,:)+ .1*(myAxis(4)-myAxis(3)), sprintf('Effective SNR: %.1f dB', dl_snrs(2,:)));
% set(h,'Color',[1 0 0])
% set(h,'FontWeight','bold')
% set(h,'FontSize',10)
% set(h,'EdgeColor',[1 0 0])
% set(h,'BackgroundColor',[1 1 1])
% hold off
% xlabel('Data Symbol Index')
% ylabel('EVM (%)');
% legend('Per-Symbol EVM','Average EVM','Location','NorthWest');
% title('EVM vs. Data Symbol Index')
% grid on
% 
% subplot(2,2,3)
% plot(100*dl_evm_mat(3,:),'o','MarkerSize',1)
% axis tight
% hold on
% plot([1 length(dl_evm_mat(3,:))], 100*[dl_aevms(3,:), dl_aevms(3,:)],'r','LineWidth',2)
% myAxis = axis;
% h = text(round(.05*length(dl_evm_mat(3,:))), 100*dl_aevms(3,:)+ .1*(myAxis(4)-myAxis(3)), sprintf('Effective SNR: %.1f dB', dl_snrs(3,:)));
% set(h,'Color',[1 0 0])
% set(h,'FontWeight','bold')
% set(h,'FontSize',10)
% set(h,'EdgeColor',[1 0 0])
% set(h,'BackgroundColor',[1 1 1])
% hold off
% xlabel('Data Symbol Index')
% ylabel('EVM (%)');
% legend('Per-Symbol EVM','Average EVM','Location','NorthWest');
% title('EVM vs. Data Symbol Index')
% grid on
% 
% subplot(2,2,4)
% plot(100*dl_evm_mat(4,:),'o','MarkerSize',1)
% axis tight
% hold on
% plot([1 length(dl_evm_mat(4,:))], 100*[dl_aevms(4,:), dl_aevms(4,:)],'r','LineWidth',2)
% myAxis = axis;
% h = text(round(.05*length(dl_evm_mat(4,:))), 100*dl_aevms(4,:)+ .1*(myAxis(4)-myAxis(3)), sprintf('Effective SNR: %.1f dB', dl_snrs(4,:)));
% set(h,'Color',[1 0 0])
% set(h,'FontWeight','bold')
% set(h,'FontSize',10)
% set(h,'EdgeColor',[1 0 0])
% set(h,'BackgroundColor',[1 1 1])
% hold off
% xlabel('Data Symbol Index')
% ylabel('EVM (%)');
% legend('Per-Symbol EVM','Average EVM','Location','NorthWest');
% title('EVM vs. Data Symbol Index')
% grid on


fprintf('\nUL Results:\n');
fprintf('===== SNRs: =====\n');

for n_ue = 1:N_UE
    fprintf('UL SNR of user %d :   %f\n', n_ue , ul_snrs(n_ue));
end

for n_ue = 1:N_UE
    fprintf('UL SE of user %d :   %f\n', n_ue , ul_se(n_ue));
end

fprintf(' Total SE is: %f\n' , ul_se_total);

fprintf('\n===== Errors: =====\n');
fprintf('Num Bits:   %d\n', N_UE * N_DATA_SYMS * log2(MOD_ORDER) );
% fprintf('UL Sym Errors:  %d (of %d total symbols)\n', ul_sym_errs, N_UE * N_DATA_SYMS);
% fprintf('UL Bit Errors:  %d (of %d total bits)\n', ul_bit_errs, N_UE * N_DATA_SYMS * log2(MOD_ORDER));

% fprintf('\n\nDL Results:\n');
% fprintf('===== SNRs: =====\n');
% 
% for n_ue = 1:N_UE
%     fprintf('DL SNR of user %d :   %f\n', n_ue , dl_snrs(n_ue));
% end
% 
% fprintf('\n===== Errors: =====\n');
% fprintf('Num Bits:   %d\n', N_UE * N_DATA_SYMS * log2(MOD_ORDER) );
% fprintf('DL Sym Errors:  %d (of %d total symbols)\n', dl_sym_errs, N_UE * N_DATA_SYMS);
% fprintf('DL Bit Errors:  %d (of %d total bits)\n', dl_bit_errs, N_UE * N_DATA_SYMS * log2(MOD_ORDER));

