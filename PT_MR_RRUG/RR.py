from tracemalloc import start
import numpy as np 
import time
import matplotlib.pyplot as plt 
from itertools import combinations
from usr_group import usr_group
from mimo_sim_ul import *
import h5py


N_UE = 16
Sche_N_UE = 4
MOD_ORDER = np.array([16,16,16,16])
max_episode_length = 400
# user_set = [0,1,2,3]
ue_history_steps = np.zeros((max_episode_length,N_UE))
ue_history = 0.01*np.ones(N_UE)

### Import data from hdf5 file #############################

channel_file = h5py.File('./16_4_16_random.hdf5','r')
H_r = np.array(channel_file.get('H_r'))
H_i = np.array(channel_file.get('H_i'))
H_i = np.transpose(H_i,(2,1,0))
H_r = np.transpose(H_r,(2,1,0))
H = np.array(H_r + 1j*H_i)
print("H shape is:", H.shape)

#[0-4] [5-9] [10-15]
#0123 1234 0234 0134 0124 

#[0,2,3,6,10] [1,4,7,8,11] [6,9,11,12,13,14,15]
#[0,4,5] [1,2,11,13] [6,14] [3,7,10] [8,9] [12,15]

for n_tti in range (0,400):
    start_time = time.time()
    H_T = H[n_tti,:,:]
    gl = usr_group(H_T)
    ug = time.time()
    print('The number of round:',n_tti)
    if n_tti % 15 == 0:
        H_matrix = np.reshape(H_T[:,0:4],(16,4))
        
        _, _, ur_se = data_process(H_matrix,4,MOD_ORDER)
        data_time = time.time()
        print("data processing time:", data_time - ug)
        ue_history[0:4] += ur_se
    elif n_tti % 15 == 1:
        H_matrix = np.reshape(H_T[:,5:9],(16,4))
        _, _, ur_se = data_process(H_matrix,4,MOD_ORDER)
        ue_history[5:9] += ur_se
    elif n_tti % 15 == 2:
        H_matrix = np.reshape(H_T[:,10:14],(16,4))
        _, _, ur_se = data_process(H_matrix,4,MOD_ORDER)
        ue_history[10:14] += ur_se
    elif n_tti % 15 == 3:
        H_matrix = np.reshape(H_T[:,1:5],(16,4))
        _, _, ur_se = data_process(H_matrix,4,MOD_ORDER)
        ue_history[1:5] += ur_se
    elif n_tti % 15 == 4:
        H_matrix = np.reshape(H_T[:,6:10],(16,4))
        _, _, ur_se = data_process(H_matrix,4,MOD_ORDER)
        ue_history[6:10] += ur_se
    elif n_tti % 15 == 5:
        H_matrix = np.reshape(H_T[:,11:15],(16,4))
        _, _, ur_se = data_process(H_matrix,4,MOD_ORDER)
        ue_history[11:15] += ur_se
    elif n_tti % 15 == 6:
        H_matrix = np.reshape(np.concatenate((H_T[:,0],H_T[:,2],H_T[:,3],H_T[:,4]),axis = 0),(16,4))
        _, _, ur_se = data_process(H_matrix,4,MOD_ORDER)
        ue_history[0] += ur_se[0]
        ue_history[2] += ur_se[1]
        ue_history[3] += ur_se[2]
        ue_history[4] += ur_se[3]
    elif n_tti % 15 == 7:
        H_matrix = np.reshape(np.concatenate((H_T[:,5],H_T[:,7],H_T[:,8],H_T[:,9]),axis = 0),(16,4))
        _, _, ur_se = data_process(H_matrix,4,MOD_ORDER)
        ue_history[5] += ur_se[0]
        ue_history[7] += ur_se[1]
        ue_history[8] += ur_se[2]
        ue_history[9] += ur_se[3]
    elif n_tti % 15 == 8:
        H_matrix = np.reshape(np.concatenate((H_T[:,12],H_T[:,13],H_T[:,14],H_T[:,15]),axis = 0),(16,4))
        _, _, ur_se = data_process(H_matrix,4,MOD_ORDER)
        ue_history[12] += ur_se[0]
        ue_history[13] += ur_se[1]
        ue_history[14] += ur_se[2]
        ue_history[15] += ur_se[3]
    elif n_tti % 15 == 9:
        H_matrix = np.reshape(np.concatenate((H_T[:,0],H_T[:,1],H_T[:,3],H_T[:,4]),axis = 0),(16,4))
        _, _, ur_se = data_process(H_matrix,4,MOD_ORDER)
        ue_history[0] += ur_se[0]
        ue_history[1] += ur_se[1]
        ue_history[3] += ur_se[2]
        ue_history[4] += ur_se[3]
    elif n_tti % 15 == 10:
        H_matrix = np.reshape(np.concatenate((H_T[:,5],H_T[:,6],H_T[:,8],H_T[:,9]),axis = 0),(16,4))
        _, _, ur_se = data_process(H_matrix,4,MOD_ORDER)
        ue_history[5] += ur_se[0]
        ue_history[6] += ur_se[1]
        ue_history[8] += ur_se[2]
        ue_history[9] += ur_se[3]
    elif n_tti % 15 == 11:
        H_matrix = np.reshape(np.concatenate((H_T[:,10],H_T[:,11],H_T[:,14],H_T[:,15]),axis = 0),(16,4))
        _, _, ur_se = data_process(H_matrix,4,MOD_ORDER)
        ue_history[10] += ur_se[0]
        ue_history[11] += ur_se[1]
        ue_history[14] += ur_se[2]
        ue_history[15] += ur_se[3]
    elif n_tti % 15 == 12:
        H_matrix = np.reshape(np.concatenate((H_T[:,0],H_T[:,1],H_T[:,2],H_T[:,4]),axis = 0),(16,4))
        _, _, ur_se = data_process(H_matrix,4,MOD_ORDER)
        ue_history[0] += ur_se[0]
        ue_history[1] += ur_se[1]
        ue_history[2] += ur_se[2]
        ue_history[4] += ur_se[3]
    elif n_tti % 15 == 13:
        H_matrix = np.reshape(np.concatenate((H_T[:,5],H_T[:,6],H_T[:,7],H_T[:,9]),axis = 0),(16,4))
        _, _, ur_se = data_process(H_matrix,4,MOD_ORDER)
        ue_history[5] += ur_se[0]
        ue_history[6] += ur_se[1]
        ue_history[7] += ur_se[2]
        ue_history[9] += ur_se[3]
    elif n_tti % 15 == 14:
        H_matrix = np.reshape(np.concatenate((H_T[:,10],H_T[:,11],H_T[:,12],H_T[:,15]),axis = 0),(16,4))
        _, _, ur_se = data_process(H_matrix,4,MOD_ORDER)
        ue_history[10] += ur_se[0]
        ue_history[11] += ur_se[1]
        ue_history[12] += ur_se[2]
        ue_history[15] += ur_se[3]
    # else:
    #     H_matrix = np.reshape(H_T[:,12:16],(16,4))
    #     _, _, ur_se = data_process(H_matrix,Sche_N_UE,MOD_ORDER)
    #     ue_history[12:16] += ur_se
    
    print(ue_history)
    
    # ue_history[(n_tti%16)] += ul_se_total
    ue_history_steps[n_tti,:] = ue_history
    
    end_time = time.time()

    running_time = (end_time - start_time)

    print('Running Time is:', running_time)

with h5py.File("../plot_data/US_rdm_2.hdf5", "w") as data_file:
    data_file.create_dataset("history", data=ue_history_steps)


print('RR data is stored\n')
