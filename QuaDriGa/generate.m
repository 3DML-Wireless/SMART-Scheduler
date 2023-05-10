% addpath('/home/qing/Downloads/Quadriga/quadriga_src')
% more off
close all
clear all

numDrops = 1;                       % Number of random user drops to simulate
sequenceLength = 400;                % Number of steps in a track to simulate for each drop

K = 8;                              % Number of users
disp('8 users')
centerFrequency = 3.6e9;            % Center frequency in Hz
bandwidth = 20e6;                     % Bandwidth in Hz
numSubcarriers = 600;               % Number of sub-carriers
subSampling = 1;                     % Only take every subsampling's sub-carriers

antennaHeight = 25;                  % Antenna height of the bse station in m
antennaSpacing = 1/2;                % Antenna spacing in multiples of the wave length
M_V = 2;                             % Number of vertical antenna elements
M_H = 2;                             % Number of horizontal antenn elements
M = 2*M_V*M_H;                       % Total number of antennas (factor 2 due to dual polarization)

minDistance = 10;                    % Minimum distance from the base station
maxDistance = 100;                   % Maximum distance from the base station
userHeight = 1.5;                    % Antenna height of the users
sectorAngle = 360;                    % Width of the simulated cell sector in deg
sectorAngleRad = sectorAngle/180*pi; % Width of the simulated cell sector in radians

lambda = 3e8/centerFrequency;
% speed_km_h = 30
% speed_m_s = speed_km_h*1000/3600
% coherenceTime = lambda/4/speed_m_s
% num_symbols_coherence_time = coherenceTime/1e-3*14*subSampling

% Scenario
s = qd_simulation_parameters;                           % Set up simulation parameters
s.show_progress_bars = 0;                               % Disable progress bars
s.center_frequency = centerFrequency;                   % Set center frequency
s.sample_density = 1;                                   % 2 samples per half-wavelength
s.use_absolute_delays = 0;                              % Include delay of the LOS path

% Layout
l = qd_layout(s);                                       % Create new QuaDRiGa layout

% Base station
l.no_tx = 1;
l.tx_position(3) = antennaHeight;
l.tx_array = qd_arrayant('3gpp-3d', M_V, M_H, centerFrequency, 3, 0, antennaSpacing);

for n=1:M_V
    for nn=1:M_H
        indeces = (n-1)*M_H+nn;
        l.tx_array.element_position(1,indeces) =  (nn)*antennaSpacing*lambda  - lambda/4 - M_V/2*antennaSpacing*lambda;
        l.tx_array.element_position(2,indeces) = 0;
        l.tx_array.element_position(3,indeces) = (n)*antennaSpacing*lambda - lambda/4 - M_H/2*antennaSpacing*lambda ;
    end
end




% + antennaHeight
% Users
l.no_rx = K;                                            % Number of users
l.rx_array = qd_arrayant( 'omni' );                     % Omnidirectional MT antenna

% Update Map
l.set_scenario('3GPP_3D_UMi_LOS');

par.minDistance = minDistance;
par.maxDistance = maxDistance;
par.sectorAngleRad = sectorAngleRad;
par.bandwidth = bandwidth;
par.numSubcarriers = numSubcarriers;
par.subSampling = subSampling;
par.sequenceLength = sequenceLength;
par.s=s;

params = cell(1,numDrops);
for n=1:numDrops
    params{1,n} = par;
    params{1,n}.l = l.copy;
end


h = cell(1,numDrops);
for n=1:numDrops
    n
    h(1,n) = genChannelDrop(params{1,n});
end
H = cell2mat(h');
clear h


% h = cell(1,numDrops);
% for n=1:numDrops
%     n
%     h(1,n) = gen_high_corr_64(params{1,n});
% end
% H = cell2mat(h');
% clear h
% size(H)

   

    % virtualize
    
% h = cell(1,numDrops);
% for n=1:numDrops
%     n
%     h(1,n) = gen_low_corr(params{1,n});
% end
% H = cell2mat(h');
% clear h

%%%%% Matrix Normalization %%%%% (MMNET PAGE 6)



H = H ./ sqrt(sum(abs(H).^2,[3 4])./ prod(size(H,2:4)));
% size(H)

H_c = zeros(50,400,8,8);

% H = squeeze(mean(H,4));
for i = 1:50
    H_c(i,:,:,:) = reshape(squeeze(mean(H(:,:,:,(i-1)*12+1:i*12,:),4)),400,M,K);
%     H_c(i,:,:,:) = H_c(i,:,:,:,:),400,M,K);
end

size(H_c)
% H1 = squeeze(mean(H(:,:,:,1:12,:),4));
% H2 = squeeze(mean(H(:,:,:,13:24,:),4));


%H = reshape(H,400,M,K);
% H1 = reshape(H1,400,M,K);
% H2 = reshape(H2,400,M,K);
% % M is BS number, K is user number, 400 is TTI number
% for i = 2:400
%     H1(i,:,:) = H1(1,:,:);
%     H2(i,:,:) = H2(1,:,:);
% end 
% 
% size(H)


% SNR = 10* log10 (squeeze(min( abs ( H (1, : ,:) ) .^2 ,[],2))./(1e-2))
% SNR = 10* log10 (squeeze(min( abs ( H_c (1,40, : ,:) ) .^2 ,[],2))./(1e-2))
% SNR = 10* log10 (squeeze(min( abs ( H_c (2,40, : ,:) ) .^2 ,[],2))./(1e-2))
% SNR = 10* log10 (squeeze(min( abs ( H1 (200, : ,:) ) .^2 ,[],2))./(1e-2))
% SNR = 10* log10 (squeeze(min( abs ( H2 (200, : ,:) ) .^2 ,[],2))./(1e-2))
% SNR_88 = 10* log10 (squeeze(sum( abs ( H (88, : ,:) ) .^2 ,2))./(1e-2))
% SNR_300 = 10* log10 (squeeze(sum( abs ( H (300, : ,:) ) .^2 ,2))./(1e-2))


% Pow = sum(squeeze(sum( abs ( H (1, : ,:) ) .^2 ,3)));
% Pow_db = 10* log10 (Pow)
% Pow_88 = sum(squeeze(sum( abs ( H (88, : ,:) ) .^2 ,3)));
% Pow_db88 = 10* log10 (Pow_88)





% for i = 1:400
%     for j = 1:12
%         H(i,:,j) = 0.1*H(i,:,j);
%     end
%     for j = 13:16
%         H(i,:,j) = 2*H(i,:,j);
%     end
% end

% 
% CN = zeros(700);
% R = zeros(700);
% 
% for i = 1:700
%     CN(i) = cond(squeeze(H(i,:,:)),2);
%     R(i) = rank(squeeze(H(i,:,:)));
% end
% H_matrix = zeros(400,M,K);
% for i = 1:400
%     H_matrix(i,:,:) = H;
% end
% size(H_matrix)

% H_r1 = real(H1);
% H_i1 = imag(H1);
% H_r2 = real(H2);
% H_i2 = imag(H2);

H_rc = real(H_c);
H_ic = imag(H_c);

size(H_rc)
size(H_ic)

clear H
hdf5write('./8_8_2RB_mob_50.hdf5', 'H_rc', H_rc, 'H_ic',H_ic);
