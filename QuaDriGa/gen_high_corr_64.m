function H = gen_high_corr_64(par)
%     %Create tracks
    for i=1:par.l.no_rx
        name = par.l.track(1,i).name;
        par.l.track(1,i) = qd_track('linear', 0.25, 3*pi/4); % static
        par.l.track(1,i).name = name;
        par.l.track(1,i).set_speed (0.1/3.6);
        par.l.track(1,i).interpolate( 'time', 10e-3, [], [], 1);
        par.l.track(1,i).scenario = '3GPP_3D_UMi_LOS';
    end

    
    % Interpolate positions to get spacial sample



    distances = sqrt(rand(1,par.l.no_rx)*(par.maxDistance^2 - par.minDistance^2) + par.minDistance^2);
    angles = (2*rand(1,par.l.no_rx)-1)*par.sectorAngleRad;
    par.l.rx_position = [cos(angles).*distances; sin(angles).*distances; 1.5.*ones(1,par.l.no_rx)];



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
%         h = squeeze(H_raw(k).fr(par.bandwidth, par.numSubcarriers));
        H(1,k,:,:,:) = h(:,1:par.subSampling:end,:);
    end
    H = {H};
end
