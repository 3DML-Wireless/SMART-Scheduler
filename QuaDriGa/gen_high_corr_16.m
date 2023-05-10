function H = gen_high_corr_16(par)
%     %Create tracks
    for i=1:par.l.no_rx
        name = par.l.track(1,i).name;
        par.l.track(1,i) = qd_track('linear', 0,0);
        par.l.track(1,i).name = name;
        par.l.track(1,i).scenario = 'LOSonly';
        par.l.rx_position(:,1) = [50; 50; 1.5];
        par.l.rx_position(:,2) = [100; 100; 1.5];
        par.l.rx_position(:,3) = [103; 103; 1.5];
        par.l.rx_position(:,4) = [105; 105; 1.5];
        par.l.rx_position(:,5) = [106; 106; 1.5];
    end

    
    % Interpolate positions to get spacial sample


%     for i=1:1
%         name = par.l.track(1,i).name;
%         par.l.track(1,i) = qd_track('linear', 0.25, 3*pi/4);
%         par.l.track(1,i).name = name;
%         par.l.track(1,i).set_speed (0.1/3.6);
%         par.l.track(1,i).interpolate( 'time', 10e-3, [], [], 1); 
%         par.l.track(1,i).scenario = '3 GPP_38 .901 _UMa';
%         par.l.rx_position(:,1) = [10; -10; 1.5];
% %         par.l.rx_position(:,2) = [13; -13; 1.5];
% %         par.l.rx_position(:,3) = [15; -15; 1.5];
% %         par.l.rx_position(:,4) = [16; -16; 1.5];
%         
%     end
% 
% 
%     for i=2:5
%         name = par.l.track(1,i).name;
%         par.l.track(1,i) = qd_track('linear', 0.25, 3*pi/4);
%         par.l.track(1,i).name = name;
%         par.l.track(1,i).set_speed (0.1/3.6);
%         par.l.track(1,i).interpolate( 'time', 10e-3, [], [], 1); 
%         par.l.track(1,i).scenario = '3 GPP_38 .901 _UMa';
%         par.l.rx_position(:,2) = [1000000; 1000; 1.5];
%         par.l.rx_position(:,3) = [1003000; 1003; 1.5];
%         par.l.rx_position(:,4) = [1005000; 1005; 1.5];
%         par.l.rx_position(:,5) = [1006000; 1006; 1.5];
%     end
%     
% 
%     for i=9:12
%         name = par.l.track(1,i).name;
%         par.l.track(1,i) = qd_track('linear', 0.25, 3*pi/4);
%         par.l.track(1,i).name = name;
%         par.l.track(1,i).set_speed (0.1/3.6);
%         par.l.track(1,i).interpolate( 'time', 10e-3, [], [], 1); 
%         par.l.track(1,i).scenario = '3GPP_3D_UMi_LOS';
%         par.l.rx_position(:,9) = [-1000; 1000; 1.5];
%         par.l.rx_position(:,10) = [-1003; 1003; 1.5];
%         par.l.rx_position(:,11) = [-1005; 1005; 1.5];
%         par.l.rx_position(:,12) = [-1006; 1006; 1.5];
%         
%     end


%     for i=13:par.l.no_rx
%         name = par.l.track(1,i).name;
%         par.l.track(1,i) = qd_track('linear', 0.25, 3*pi/4);
%         par.l.track(1,i).name = name;
%         par.l.track(1,i).set_speed (0.1/3.6);
%         par.l.track(1,i).interpolate( 'time', 10e-3, [], [], 1); 
%         par.l.track(1,i).scenario = '3GPP_3D_UMi_LOS';
%         par.l.rx_position(:,13) = [-1000; -1000; 1.5];
%         par.l.rx_position(:,14) = [-1003; -1003; 1.5];
%         par.l.rx_position(:,15) = [-1005; -1005; 1.5];
%         par.l.rx_position(:,16) = [-1006; -1006; 1.5];
%         
%     end


%     for i=1:8
%         name = par.l.track(1,i).name;
%         par.l.track(1,i) = qd_track('linear', 0.25, 3*pi/4);
%         par.l.track(1,i).name = name;
%         par.l.track(1,i).set_speed (0.1/3.6);
%         par.l.track(1,i).interpolate( 'time', 10e-3, [], [], 1); 
%         par.l.track(1,i).scenario = '3GPP_3D_UMi_LOS';
%         par.l.rx_position(:,1) = [100; -100; 1.5];
%         par.l.rx_position(:,2) = [103; -103; 1.5];
%         par.l.rx_position(:,3) = [105; -105; 1.5];
%         par.l.rx_position(:,4) = [106; -106; 1.5];
%         par.l.rx_position(:,5) = [100; -103; 1.5];
%         par.l.rx_position(:,6) = [103; -100; 1.5];
%         par.l.rx_position(:,7) = [105; -103; 1.5];
%         par.l.rx_position(:,8) = [105; -106; 1.5];
%     end
% 
% 
%     for i=1:par.l.no_rx
%         name = par.l.track(1,i).name;
%         par.l.track(1,i) = qd_track('linear', 0.25, 3*pi/4);
%         par.l.track(1,i).name = name;
%         par.l.track(1,i).set_speed (0.1/3.6);
%         par.l.track(1,i).interpolate( 'time', 10e-3, [], [], 1); 
%         par.l.track(1,i).scenario = '3GPP_3D_UMi_LOS';
%         par.l.rx_position(:,9) = [-100; 100; 1.5];
%         par.l.rx_position(:,10) = [-103; 103; 1.5];
%         par.l.rx_position(:,11) = [-105; 105; 1.5];
%         par.l.rx_position(:,12) = [-106; 106; 1.5];
%         par.l.rx_position(:,13) = [-100; 103; 1.5];
%         par.l.rx_position(:,14) = [-103; 100; 1.5];
%         par.l.rx_position(:,15) = [-105; 105; 1.5];
%         par.l.rx_position(:,16) = [-106; 105; 1.5];
%     end
    
%     distances = sqrt(rand(1,par.l.no_rx)*(par.maxDistance^2 - par.minDistance^2) + par.minDistance^2);
%     angles = (2*rand(1,par.l.no_rx)-1)*par.sectorAngleRad;
%     par.l.rx_position = [cos(angles).*distances; sin(angles).*distances; 1.5.*ones(1,par.l.no_rx)];



    for i=1:par.l.no_rx
        a = par.l.track(1,i).initial_position+par.l.track(1,i).positions;
        if sum(abs(atan(a(2,:)./a(1,:))) > par.sectorAngleRad)
            disp('Out of sector angle')
            i
        end
        if sum(sqrt(a(1,:).^2+a(2,:).^2) > par.maxDistance)
            disp('Out of range r')
            i
        end
    end
    
%     sample_distance = 5;                                    % One pixel every 5 m
%     x_min           = -50;                                  % Area to be samples in [m]
%     x_max           = 500;
%     y_min           = -500;
%     y_max           = 500;
%     rx_height       = 1.5;                                  % Mobile terminal height in [m]
%     tx_power        = 30;                                   % Tx-power in [dBm] per antenna element
%     i_freq          = 1;
% 
%     [ map, x_coords, y_coords] = par.l.power_map( '3GPP_38.901_UMa_LOS', 'quick',...
%     sample_distance, x_min, x_max, y_min, y_max, rx_height, tx_power, i_freq );
%     P = 10*log10( sum( map{1}, 4 ) );
%     par.l.visualize ([] ,[] ,0) ;
%     hold on
%     imagesc ( x_coords , y_coords , P ) ;
%     hold off
%     axis ([ -50 500 -500 500])
%     caxis ( max ( P (:) ) + [ -20 0] )
%     colmap = colormap ;
%     colormap ( colmap *0.5 + 0.5 ) ;
%     set ( gca , 'layer ' , 'top')


    set (0 , 'DefaultFigurePaperSize',[14.5 7.7])
    par.l.visualize ([] ,[] ,0) ;
    view (-33 , 45) ;
    hold on

    
    % Get channel impulse reponses
    H_raw = par.l.get_channels();
    
    % Get channels on sub-carriers
    H = zeros(1,par.l.no_rx, par.l.tx_array.no_elements, par.numSubcarriers/par.subSampling, par.sequenceLength);
    for k=1:par.l.no_rx
        h = squeeze(H_raw(k).fr(par.bandwidth, par.numSubcarriers, 1:par.sequenceLength));
        H(1,k,:,:,:) = h(:,1:par.subSampling:end,:);
    end
    H = {H};
end
