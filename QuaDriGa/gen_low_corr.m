function H = gen_low_corr(par)
    %Create tracks
%     for i=1:par.l.no_rx
%         name = par.l.track(1,i).name;
%         par.l.track(1,i) = qd_track('linear', 1/par.s.samples_per_meter*(par.sequenceLength-1));
%         par.l.track(1,i).name = name;
%         par.l.track(1,i).scenario = '3GPP_3D_UMi_LOS';
%     end

    for i=1:par.l.no_rx
        name = par.l.track(1,i).name;
        par.l.track(1,i) = qd_track('linear', 0.25, 3*pi/4);
        par.l.track(1,i).name = name;
        par.l.track(1,i).set_speed (0.1/3.6);
        par.l.track(1,i).interpolate( 'time', 10e-3, [], [], 1); 
        par.l.track(1,i).scenario = '3GPP_3D_UMi_LOS';
    end

    % Add random positions (All UEs in a circle (r=2) around (20,20))
    % distances = rand(1,par.l.no_rx)*2;
    % angles = rand(1,par.l.no_rx)*2*pi;
    par.l.rx_position = [[0,5000,-5000,0]; [50,5000,5000,-5000]; 1.5.*ones(1,par.l.no_rx)];

    % Interpolate positions to get spacial samples
    
    
%     for i=1:par.l.no_rx
%         a = par.l.track(1,i).initial_position+par.l.track(1,i).positions;
%         if sum(abs(atan(a(2,:)./a(1,:))) > par.sectorAngleRad)
%             disp('Out of sector angle')
%             i
%         end
%         if sum(sqrt(a(1,:).^2+a(2,:).^2) > par.maxDistance)
%             disp('Out of range r')
%             i
%         end
%     end
    
    % virtualize
    
    set (0 , 'DefaultFigurePaperSize',[14.5 7.7])
    par.l.visualize ([] ,[] ,0) ;
    view (-33 , 45) ;
    
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
