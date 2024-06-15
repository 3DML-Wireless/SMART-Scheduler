#########################################
#              Import Library           #
#########################################

import numpy as np
import numpy.matlib
import time
import math
import os
import matplotlib
import matplotlib.pyplot as plt

#########################################
#            Global Parameters          #
#########################################

# Waveform params
N_OFDM_SYMS             = 24           # Number of OFDM symbols
# MOD_ORDER               = 16           # Modulation order (2/4/16/64 = BSPK/QPSK/16-QAM/64-QAM)
TX_SCALE                = 1.0          # Scale for Tdata waveform ([0:1])

# OFDM params
SC_IND_PILOTS           = np.array([7, 21, 43, 57])                           # Pilot subcarrier indices
#print(SC_IND_PILOTS)
SC_IND_DATA             = np.r_[1:7,8:21,22:27,38:43,44:57,58:64]     # Data subcarrier indices
#print(SC_IND_DATA)
N_SC                    = 64                                     # Number of subcarriers
# CP_LEN                  = 16                                    # Cyclic prefidata length
N_DATA_SYMS             = N_OFDM_SYMS * len(SC_IND_DATA)     # Number of data symbols (one per data-bearing subcarrier per OFDM symbol)

SAMP_FREQ               = 20e6

# Massive-MIMO params
# N_UE                    = 4
N_BS_ANT                = 64               # N_BS_ANT >> N_UE
# N_UPLINK_SYMBOLS        = N_OFDM_SYMS
N_0                     = 1e-2
H_var                   = 0.1


# LTS for CFO and channel estimation
lts_f = np.array([0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1])
# lts_t = np.fft.ifft(lts_f, 64)
#print(lts_t)



#########################################
#      Modulation and Demodulation      #
#########################################

modvec_bpsk   =  (1/np.sqrt(2))  * np.array([-1, 1]) # and QPSK
modvec_16qam  =  (1/np.sqrt(10)) * np.array([-3, -1, +3, +1])
modvec_64qam  =  (1/np.sqrt(43)) * np.array([-7, -5, -1, -3, +7, +5, +1, +3])


def modulation (mod_order,data):
    
    if (mod_order == 2): #BPSK
        return complex(modvec_bpsk[data],0) # data = 0/1
    elif (mod_order == 4): #QPSK
        return complex(modvec_bpsk[data>>1],modvec_bpsk[np.mod(data,2)])
    elif (mod_order == 16): #16-QAM
        return complex(modvec_16qam[data>>2],modvec_16qam[np.mod(data,4)])
    elif (mod_order == 64): #64-QAM
        return complex(modvec_64qam[data>>3],modvec_64qam[np.mod(data,8)])

def demodulation (mod_order, data):

    if (mod_order == 2): #BPSK
        return float(np.real(data)>0) # data = 0/1
    elif (mod_order == 4): #QPSK
        return float(2*(np.real(data)>0) + 1*(np.imag(data)>0))
    elif (mod_order == 16): #16-QAM
        return float((8*(np.real(data)>0)) + (4*(abs(np.real(data))<0.6325)) + (2*(np.imag(data)>0)) + (1*(abs(np.imag(data))<0.6325)))
    elif (mod_order == 64): #64-QAM
        return float((32*(np.real(data)>0)) + (16*(abs(np.real(data))<0.6172)) + (8*((abs(np.real(data))<(0.9258))and((abs(np.real(data))>(0.3086))))) + (4*(np.imag(data)>0)) + (2*(abs(np.imag(data))<0.6172)) + (1*((abs(np.imag(data))<(0.9258))and((abs(np.imag(data))>(0.3086))))))

## H:(N_BS,N_UE), N_UE scalar, MOD_ORDER:(N_UE,)
def data_process (H, N_UE, MOD_ORDER):

    pilot_in_mat = np.zeros((N_UE, N_SC, N_UE));
    for i in range (0, N_UE):
        pilot_in_mat [i, :, i] = lts_f;

    lts_f_mat = np.zeros((N_BS_ANT, N_SC, N_UE));
    for i in range (0, N_UE):
        lts_f_mat[:, :, i] = numpy.matlib.repmat(lts_f, N_BS_ANT, 1);

    ## Uplink

    # Generate a payload of random integers
    tx_ul_data = np.zeros((N_UE, N_DATA_SYMS),dtype='int')
    for n_ue in range (0,N_UE):
        tx_ul_data[n_ue,:] = np.random.randint(low = 0, high = MOD_ORDER[n_ue], size=(1, N_DATA_SYMS))

    # Map the data values on to complex symbols
    tx_ul_syms = np.zeros((N_UE, N_DATA_SYMS),dtype='complex')
    vec_mod = np.vectorize(modulation)
    for n_ue in range (0,N_UE):
        tx_ul_syms[n_ue,:] = vec_mod(MOD_ORDER[n_ue], tx_ul_data[n_ue,:])

    #print(tx_ul_syms.shape)

    # Reshape the symbol vector to a matrix with one column per OFDM symbol
    tx_ul_syms_mat = np.reshape(tx_ul_syms, (N_UE, len(SC_IND_DATA), N_OFDM_SYMS))

    # Define the pilot tone values as BPSK symbols
    pt_pilots = np.transpose(np.array([[1, 1, -1, 1]]))

    # Repeat the pilots across all OFDM symbols
    pt_pilots_mat = np.zeros((N_UE, 4, N_OFDM_SYMS),dtype= 'complex')

    for i in range (0,N_UE):
        pt_pilots_mat[i,:,:] = numpy.matlib.repmat(pt_pilots, 1, N_OFDM_SYMS)

    ## IFFT

    # Construct the IFFT input matrix
    data_in_mat = np.zeros((N_UE, N_SC, N_OFDM_SYMS),dtype='complex')

    # Insert the data and pilot values; other subcarriers will remain at 0
    data_in_mat[:, SC_IND_DATA, :] = tx_ul_syms_mat
    data_in_mat[:, SC_IND_PILOTS, :] = pt_pilots_mat

    # Apply hardware phase distortion
    # ifft_in_hw_mat = np.multiply(data_in_mat, np.transpose(np.tile(ul_tx_hw_phase, (N_OFDM_SYMS,1,1)),(1,2,0)))

    # #Perform the IFFT
    # tx_payload_mat = np.fft.ifft(ifft_in_hw_mat, N_SC, 1)

    # # Insert the cyclic prefix
    # if(CP_LEN > 0):
    #     tx_cp = tx_payload_mat[:, -CP_LEN : , :]
    #     tx_payload_mat = np.concatenate((tx_cp, tx_payload_mat),axis=1); #[tx_cp; tx_payload_mat]

    tx_mat_f = np.concatenate((pilot_in_mat, data_in_mat),axis=2);
    # Reshape to a vector
    tx_payload_vec = np.reshape(tx_mat_f, (N_UE, -1))
    # tx_pilot_vec = np.zeros([N_UE, SYM_LEN * (N_UE+1)],dtype= 'complex'); # additional pilot as noise
    # for i in range (0,N_UE):
    #     lts_t = np.fft.ifft(np.multiply(lts_f, ul_tx_hw_phase[i, :]), 64)
    #     tx_pilot_vec[i, i*SYM_LEN:((i+1)*SYM_LEN)] = np.concatenate((lts_t[63-CP_LEN+1:], lts_t))


    # # Construct the full time-domain OFDM waveform
    # tx_vec = np.concatenate((tx_pilot_vec, tx_payload_vec),axis=1)
    # #print(tx_vec.shape)
    # tx_vec_air = np.divide((TX_SCALE * tx_vec), numpy.matlib.repmat(np.reshape(np.amax(np.abs(tx_vec),axis=1),(-1,1)), 1, tx_vec.shape[1]))

    # UL noise matrix
    Z_mat = np.sqrt(N_0/2) * ( np.random.random((N_BS_ANT,tx_payload_vec.shape[1])) + 1j*np.random.random((N_BS_ANT,tx_payload_vec.shape[1])))

    # H = np.sqrt(H_var/2) * ( np.random.random((N_BS_ANT, N_UE)) + 1j*np.random.random((N_BS_ANT, N_UE)))
    rx_payload_vec = np.matmul(H, tx_payload_vec) + Z_mat
    rx_mat_f = np.reshape(rx_payload_vec, (N_BS_ANT, N_SC, N_UE + N_OFDM_SYMS))
    # rx_vec_air = np.matmul(H, tx_vec_air) + Z_mat

    # rx_pilot_vec = np.zeros((N_BS_ANT, N_SC, N_UE),dtype='complex')
    # for i in range (0,N_UE):
    #     rx_pilot_vec[:, :, i] = rx_vec_air[:, i*SYM_LEN+CP_LEN:(i+1)*SYM_LEN]


    # lts_f_mat = np.zeros((N_BS_ANT, N_SC, N_UE),dtype='complex')
    # for i in range (0,N_UE):
    #     lts_f_mat[:, :, i] = numpy.matlib.repmat(lts_f, N_BS_ANT, 1)


    csi_mat = np.multiply(rx_mat_f[:, :, 0:N_UE], lts_f_mat)
    #print(csi_mat.shape)
    fft_out_mat = rx_mat_f[:, :, N_UE:];

    # rx_payload_vec=rx_vec_air[:, (N_UE+1)*SYM_LEN:]
    # rx_payload_mat = np.reshape(rx_payload_vec, (N_BS_ANT, SYM_LEN, N_OFDM_SYMS)) # first two are preamble
    # rx_payload_mat_noCP = rx_payload_mat[:, CP_LEN: , :]
    # fft_out_mat = np.multiply(np.fft.fft(rx_payload_mat_noCP, N_SC, 1), np.transpose(np.tile(ul_rx_hw_phase, (N_OFDM_SYMS,1,1)),(1,2,0)))



    # precoding_mat = np.zeros((N_BS_ANT, N_SC, N_UE),dtype='complex')
    demult_mat = np.zeros((N_UE, N_SC, N_OFDM_SYMS),dtype='complex')
    sc_csi_mat = np.zeros((N_BS_ANT, N_UE),dtype='complex')



    for j in range (0,N_SC):
        sc_csi_mat = csi_mat[:, j, :]
        zf_mat = np.linalg.pinv(sc_csi_mat)   # ZF
        demult_mat[:, j, :] = np.matmul(zf_mat, np.squeeze(fft_out_mat[:, j, :]))
        # dl_zf_mat = np.linalg.pinv(np.matmul(np.diag(calib_mat[:, i]), sc_csi_mat))
        # #print(dl_zf_mat.shape)
        # precoding_mat[:, j, :] = np.transpose((dl_zf_mat))      # zf_mat.';

    # pilots_f_mat = demult_mat[:, SC_IND_PILOTS, :]
    # pilots_f_mat_comp = np.multiply(pilots_f_mat, pt_pilots_mat)
    # pilot_phase_err = np.squeeze(np.angle(np.mean(pilots_f_mat_comp, 1)))

    # pilot_phase_corr = np.zeros((N_UE, N_SC, N_OFDM_SYMS),dtype='complex')
    # for i in range (0,N_SC):
    #     pilot_phase_corr[:,i,:] = np.exp(-1j*pilot_phase_err)


    # # Apply the pilot phase correction per symbol
    # demult_pc_mat = np.multiply(demult_mat, pilot_phase_corr)
    payload_syms_mat = demult_mat[:, SC_IND_DATA, :]
    payload_syms_mat = np.reshape(payload_syms_mat, (N_UE, -1))

    # Demodulate uplink
    # rx_ul_syms = payload_syms_mat
    # rx_ul_data = np.zeros(rx_ul_syms.shape)

    # vec_demod = np.vectorize(demodulation)

    # for n_ue in range (0,N_UE):
    #     rx_ul_data[n_ue,:] = vec_demod(MOD_ORDER[n_ue], rx_ul_syms[n_ue,:])
    # #print(rx_ul_data)

    # ## Calculate UL Rx stats

    # ul_sym_errs = np.sum(np.sum(tx_ul_data != rx_ul_data)) # errors per user
    # #print(ul_sym_errs)
    #ul_bit_errs = len(np.argwhere(np.binary_repr(np.bitwise_xor(tx_ul_data, rx_ul_data), 8) == '1'))

    tx_ul_syms_vecs = np.reshape(tx_ul_syms_mat, (N_UE, -1))
    ul_evm_mat = np.square(np.abs(payload_syms_mat - tx_ul_syms_vecs))
    ul_aevms = np.mean(ul_evm_mat, 1)
    ul_snrs = 10*np.log10(1 / ul_aevms)

    ul_snr_discount = np.zeros(N_UE)
    for n_ue in range (0,N_UE):
        if (MOD_ORDER[n_ue] == 4):
            ul_snr_discount[n_ue] = ul_snrs[n_ue] + 10*np.log10(4)
        elif (MOD_ORDER[n_ue] == 16):
            ul_snr_discount[n_ue] = ul_snrs[n_ue]
        elif (MOD_ORDER[n_ue] == 64):
            ul_snr_discount[n_ue] = ul_snrs[n_ue] - 10*np.log10(4)
        ul_snrs[n_ue] = ul_snrs[n_ue] if ul_snrs[n_ue] > 0 else 0

    ## Spectrual Efficiency
    ## SUM(min(log2(1+SNR),MOD_ORDER))
    ul_se = np.zeros(N_UE)
    for n_ue in range (0,N_UE):
        # ul_se[n_ue] = np.min(np.array([np.log2(1+ul_snrs[n_ue]),np.log2(MOD_ORDER[n_ue])]))
        ul_se[n_ue] = np.log2(1+10**(ul_snrs[n_ue]/10)) * np.log2(MOD_ORDER[n_ue])
    ul_se_total = np.sum(ul_se)





    ## Plots:
    # UL (first 4 UEs)

    # plt.figure(figsize=(10, 8), dpi=80); 

    # plt.subplot(221)
    # plt.plot(payload_syms_mat[0, :].real,payload_syms_mat[0, :].imag,'ro', ms=1.0)
    # plt.axis('square')
    # plt.axis([-1.5, 1.5, -1.5, 1.5])
    # plt.grid()


    # plt.plot(tx_ul_syms[0, :].real,tx_ul_syms[0, :].imag,'bo',markerfacecolor="None")
    # plt.title('Uplink Tx and Rx Constellations')
    # plt.legend(["Rx","Tx"], loc = "upper right")


    # plt.subplot(222)
    # plt.plot(payload_syms_mat[1, :].real,payload_syms_mat[1, :].imag,'ro', ms=1.0)
    # plt.axis('square')
    # plt.axis([-1.5, 1.5, -1.5, 1.5])
    # plt.grid()


    # plt.plot(tx_ul_syms[1, :].real,tx_ul_syms[1, :].imag,'bo',markerfacecolor="None")
    # plt.title('Uplink Tx and Rx Constellations')
    # plt.legend(["Rx","Tx"], loc = "upper right")


    # plt.subplot(223)
    # plt.plot(payload_syms_mat[2, :].real,payload_syms_mat[2, :].imag,'ro', ms=1.0)
    # plt.axis('square')
    # plt.axis([-1.5, 1.5, -1.5, 1.5])
    # plt.grid()


    # plt.plot(tx_ul_syms[2, :].real,tx_ul_syms[2, :].imag,'bo',markerfacecolor="None")
    # plt.title('Uplink Tx and Rx Constellations')
    # plt.legend(["Rx","Tx"], loc = "upper right")


    # plt.subplot(224)
    # plt.plot(payload_syms_mat[3, :].real,payload_syms_mat[3, :].imag,'ro', ms=1.0)
    # plt.axis('square')
    # plt.axis([-1.5, 1.5, -1.5, 1.5])
    # plt.grid()


    # plt.plot(tx_ul_syms[3, :].real,tx_ul_syms[3, :].imag,'bo',markerfacecolor="None")
    # plt.title('Uplink Tx and Rx Constellations')
    # plt.legend(["Rx","Tx"], loc = "upper right")
    # plt.show()
    # # plt.show(block = False)


    # ## Plot EVM
    
    # plt.figure(figsize=(10, 10), dpi=80); 

    # plt.subplot(221)
    # plt.plot((100*ul_evm_mat[0,:]),'o', ms=1)
    # plt.axis('tight')


    # plt.axhline(y = (100*ul_aevms[0]),color = 'r', linewidth=2)
    # plt.title('Downlink Rx Stats')

    # plt.text(.1*len(ul_evm_mat[0,:]),125*ul_aevms[0],'Effective SNR:{:.1f} dB'.format(ul_snrs[0]), verticalalignment='bottom',horizontalalignment='left',fontweight = 'bold',fontsize = 15, color = 'red',bbox= {'facecolor': 'white','alpha': 0.7} )


    # plt.xlabel('Data Symbol Index')
    # plt.ylabel('EVM (%)')
    # plt.legend(['Per-Symbol EVM','Average EVM'], loc = 'upper left')
    # plt.title('EVM vs. Data Symbol Index',fontweight = 'bold')
    # plt.grid()


    # plt.subplot(222)
    # plt.plot((100*ul_evm_mat[1,:]),'o', ms=1)
    # plt.axis('tight')

    # plt.axhline(y = (100*ul_aevms[1]),color = 'r', linewidth=2)
    # plt.title('Downlink Rx Stats')

    # plt.text(.1*len(ul_evm_mat[1,:]),125*ul_aevms[1],'Effective SNR:{:.1f} dB'.format(ul_snrs[1]), verticalalignment='bottom',horizontalalignment='left',fontweight = 'bold',fontsize = 15, color = 'red',bbox= {'facecolor': 'white','alpha': 0.7} )


    # plt.xlabel('Data Symbol Index')
    # plt.ylabel('EVM (%)')
    # plt.legend(['Per-Symbol EVM','Average EVM'], loc = 'upper left')
    # plt.title('EVM vs. Data Symbol Index',fontweight = 'bold')
    # plt.grid()


    # plt.subplot(223)
    # plt.plot((100*ul_evm_mat[2,:]),'o', ms=1)
    # plt.axis('tight')

    # plt.axhline(y = (100*ul_aevms[2]),color = 'r', linewidth=2)
    # plt.title('Downlink Rx Stats')

    # plt.text(.1*len(ul_evm_mat[2,:]),125*ul_aevms[2],'Effective SNR:{:.1f} dB'.format(ul_snrs[2]), verticalalignment='bottom',horizontalalignment='left',fontweight = 'bold',fontsize = 15, color = 'red',bbox= {'facecolor': 'white','alpha': 0.7} )


    # plt.xlabel('Data Symbol Index')
    # plt.ylabel('EVM (%)')
    # plt.legend(['Per-Symbol EVM','Average EVM'], loc = 'upper left')
    # plt.title('EVM vs. Data Symbol Index',fontweight = 'bold')
    # plt.grid()


    # plt.subplot(224)
    # plt.plot((100*ul_evm_mat[3,:]),'o', ms=1)
    # plt.axis('tight')

    # plt.axhline(y = (100*ul_aevms[3]),color = 'r', linewidth=2)
    # plt.title('Downlink Rx Stats')

    # plt.text(.1*len(ul_evm_mat[3,:]),125*ul_aevms[3],'Effective SNR:{:.1f} dB'.format(ul_snrs[3]), verticalalignment='bottom',horizontalalignment='left',fontweight = 'bold',fontsize = 15, color = 'red',bbox= {'facecolor': 'white','alpha': 0.7} )


    # plt.xlabel('Data Symbol Index')
    # plt.ylabel('EVM (%)')
    # plt.legend(['Per-Symbol EVM','Average EVM'], loc = 'upper left')
    # plt.title('EVM vs. Data Symbol Index',fontweight = 'bold')
    # plt.grid()
    # plt.show()



    # print('\nUL Results:\n')
    # print('===== SNRs: =====\n')

    # for n_ue in range (0,N_UE):
    #     print("UL SNR of user",n_ue,":    ",ul_snrs[n_ue], "\n")

    # for n_ue in range (0,N_UE):
    #     print("UL SE of user (bits/sec/Hz)",n_ue,":    ",ul_se[n_ue], "\n")

    # print("Total SE is:", ul_se_total,"bits/sec/Hz \n")

    # print('\n===== Errors: =====\n')
    # print("Num Bits:",  N_DATA_SYMS * np.sum(np.log2(MOD_ORDER)), "\n")
    # print("UL Sym Errors:", ul_sym_errs, "(of", N_UE * N_DATA_SYMS, "total symbols)\n")

    return ul_se_total, np.min(ul_snr_discount),ul_se