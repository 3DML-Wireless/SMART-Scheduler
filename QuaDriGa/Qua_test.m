% addpath('/home/qing/Downloads/Quadriga/quadriga_src')
% more off
close all
clear all

set(0,'defaultTextFontSize', 18)                        % Default Font Size
set(0,'defaultAxesFontSize', 18)                        % Default Font Size
set(0,'defaultAxesFontName','Times')                    % Default Font Type
set(0,'defaultTextFontName','Times')                    % Default Font Type
set(0,'defaultFigurePaperPositionMode','auto')          % Default Plot position
set(0,'DefaultFigurePaperType','<custom>')              % Default Paper Type
set(0,'DefaultFigurePaperSize',[14.5 7.7])            	% Default Paper Size

s = qd_simulation_parameters;
s.center_frequency = [2.6e9, 28e9];                     % Assign two frequencies
bandwidth = 20e6;                     % Bandwidth in Hz
numSubcarriers = 52;               % Number of sub-carriers
subSampling = 1; 
sequenceLength = 1;
numDrops = 1;

l = qd_layout( s );                                     % New QuaDRiGa layout
l.tx_position = [0 0 25]';                              % 25 m BS height
l.no_rx = 5;                                          % 100 MTs

l.rx_position(:,1) = [100; 0; 1.5];
l.rx_position(:,2) = [450; 0; 1.5];
l.rx_position(:,3) = [453; 0; 1.5];
l.rx_position(:,4) = [455; 0; 1.5];
l.rx_position(:,5) = [456; 0; 1.5];

% l.randomize_rx_positions( 200, 1.5, 1.5, 0 );           % Assign random user positions
% l.rx_position(1,:) = l.rx_position(1,:) + 220;          % Place users east of the BS
indoor_rx = l.set_scenario('3GPP_38.901_UMa',[],[],0);    % Set the scenario
% l.rx_position(3,~indoor_rx) = 1.5;


a_2600_Mhz  = qd_arrayant( '3gpp-3d',  8, 1, s.center_frequency(1), 6, 8 );
l.tx_array(1,1) = a_2600_Mhz;

l.rx_array = qd_arrayant('omni'); 
sample_distance = 5;                                    % One pixel every 5 m
x_min           = -50;                                  % Area to be samples in [m]
x_max           = 550;
y_min           = -300;
y_max           = 300;
rx_height       = 1.5;                                  % Mobile terminal height in [m]
tx_power        = 30;                                   % Tx-power in [dBm] per antenna element
i_freq          = 1;                                    % Frequency index for 2.6 GHz

% Calculate the map including path-loss and antenna patterns
[ map, x_coords, y_coords] = l.power_map( '3GPP_38.901_UMa_LOS', 'quick',...
    sample_distance, x_min, x_max, y_min, y_max, rx_height, tx_power, i_freq );
P_db = 10*log10( sum( map{1}, 4 ) );

% Plot the results
l.visualize([],[],0);                                   % Show BS and MT positions on the map
hold on; imagesc( x_coords, y_coords, P_db ); hold off  % Plot the antenna footprint
axis([x_min,x_max,y_min,y_max]);
caxis( max(P_db(:)) + [-20 0] );                        % Color range
colmap = colormap;
colormap( colmap*0.5 + 0.5 );                           % Adjust colors to be "lighter"
set(gca,'layer','top')                                  % Show grid on top of the map
colorbar('south')
title('Received power [dBm] for 2.6 GHz band')

c = l.get_channels;
% c_size = size(c)

H = zeros(1,l.no_rx, 2, numSubcarriers);
for k=1:l.no_rx
    h = squeeze(c(k).fr(bandwidth, numSubcarriers));
    % h_size = size(h)
    H(1,k,:,:,:) = h;
end
H = {H};

h = cell(1,1);

h(1,1) = H;

H = cell2mat(h');
clear h
size(H)



H = H ./ sqrt(sum(abs(H).^2,[3 4])./ prod(size(H,2:4)));

H = permute(squeeze(mean(H,4)),[1,4,3,2]);
size(H)

H = reshape(H,1,2,5);
H(1,:,:)
% M is BS number, K is user number, 400 is TTI number

size(H)

SNR = 10* log10 (squeeze(min( abs ( H (1, : ,:) ) .^2 ,[],2))./(1e-2))
% SNR_88 = 10* log10 (squeeze(sum( abs ( H (88, : ,:) ) .^2 ,2))./(1e-2))
% SNR_300 = 10* log10 (squeeze(sum( abs ( H (300, : ,:) ) .^2 ,2))./(1e-2))


Pow = sum(squeeze(sum( abs ( H (1, : ,:) ) .^2 ,3)));
Pow_db = 10* log10 (Pow)