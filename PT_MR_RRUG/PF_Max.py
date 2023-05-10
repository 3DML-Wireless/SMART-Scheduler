import numpy as np 
import time
import matplotlib.pyplot as plt 
from itertools import combinations
from mimo_sim_ul import *
import h5py

P_N_UE = 16
N_UE = 16
max_episode_length = 400
user_set = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

actions_num = 0 # number of actions
for i in range (1,P_N_UE+1):
    actions_num += len(list(combinations(user_set,i)))


pf_ue_history = 0.01*np.ones(N_UE)
max_ue_history = 0.01*np.ones(N_UE)
PF = np.zeros(actions_num)
SE = np.zeros([max_episode_length,actions_num])
last_flag = np.zeros(N_UE)
last_SE = np.zeros(N_UE)
# pf_ur_min_snr = np.zeros(max_episode_length)
pf_ue_history_steps = np.zeros((max_episode_length,N_UE))
max_ue_history_steps = np.zeros((max_episode_length,N_UE))

# pre_data = np.zeros((400,actions_num,6))
### Import data from hdf5 file #############################

channel_file = h5py.File('./16_4_16_mobile.hdf5','r')
# se_max_ur = data_shuffled_file.get('se_max')
# se_max_ur = np.array(se_max_ur)
# print('se data type:',se_max_ur.dtype)
H_r = np.array(channel_file.get('H_r'))
H_i = np.array(channel_file.get('H_i'))
H_i = np.transpose(H_i,(2,1,0))
H_r = np.transpose(H_r,(2,1,0))
H = np.array(H_r + 1j*H_i)
print("H shape is:", H.shape)


# PF rate
ld = 0.5


# last_selection = np.zeros(3)
# last_sel_SE = np.zeros(3)

### Find UE_selection Based on action
def sel_ue(action):
    sum_before = 0
    for i in range (1,(P_N_UE+1)):
        sum_before += len(list(combinations(user_set, i)))
        if ((action+1)>sum_before):
            continue
        else:
            idx = i
            sum_before -= len(list(combinations(user_set, i)))
            ue_select = list(combinations(user_set, i))[action-sum_before]
            break
    # print('action and sel is:',action, ue_select)
    return ue_select,idx


def choose_action(t):
    PF = np.zeros(actions_num)
    ur_se_tt = np.zeros(actions_num)
    ct = 0
    # print('Current Frame is:',t)
    for ac in range (0, actions_num):
        
        ue_select, idx = sel_ue(ac)
        
        mod_select = np.ones((idx,)) *16 # 16-QAM
        st = time.time()
        ul_se_total, ur_min_snr, ur_se = data_process(np.reshape(H[t,:,ue_select],(16,-1)),idx,mod_select)
        et = time.time()
        dt = et-st
        # print("dt:",dt)
        ct += dt
        # pre_data[t,ac,0] = ul_se_total
        # pre_data[t,ac,1] = ur_min_snr
        # pre_data[t,ac,2:2+idx] = ur_se
        
        ur_se_tt[ac] = ul_se_total
        if(t>0):
            for ue in ue_select:
                if (last_flag[ue]):
                    tp_history = ld*pf_ue_history[ue]/t + (1-ld) * last_SE[ue]
                else:
                    tp_history = ld*pf_ue_history[ue]/t
                PF[ac] += ur_se[ue_select.index(ue)] / tp_history
            if ac == actions_num-1:
                # print('PF is:', PF)
                print('se total:',ur_se_tt)
    print("ct is", ct)       
    return np.argmax(PF), np.argmax(ur_se_tt)


def update_pf(t, if_pf, action):

    ue_select, idx = sel_ue(action)
    print("ue sel is:", ue_select)
    mod_select = np.ones((idx,)) *16 # 16-QAM
    ur_se_total, ur_min_snr, ur_se = data_process(np.reshape(H[t,:,ue_select],(16,-1)),idx,mod_select)

    if if_pf:
        for i in range(0,idx):
            pf_ue_history[ue_select[i]] += ur_se[i]
        
        pf_jfi = np.square((np.sum(pf_ue_history))) / (16 * np.sum(np.square(pf_ue_history)))
        print("PF UE History is:", pf_ue_history)
        print("PF JFI is:", pf_jfi,"\n")
        
        for ue in range (0,N_UE):
            if (ue in ue_select):
                last_flag[ue] = 1
                last_SE[ue] = ur_se[ue_select.index(ue)]
            else:
                last_flag[ue] = 0
                last_SE[ue] = 0
        return pf_ue_history
    else:
        for i in range(0,idx):
            max_ue_history[ue_select[i]] += ur_se[i]
        max_jfi = np.square((np.sum(max_ue_history))) / (16 * np.sum(np.square(max_ue_history)))
        print("MAX UE History is:", ur_se)
        print("MAX JFI is:", max_jfi,"\n")
        return max_ue_history





#### Starting Processing #####

start_time = time.time()

for i in range (0,max_episode_length):
    s_time = time.time()
    pf_action, max_action = choose_action(i)
    # choose_time = time.time()
    # print('Data processing time is:',choose_time - s_time)
    pf_ue_history_steps[i] = update_pf(i, 1, pf_action)
    max_ue_history_steps[i] = update_pf(i, 0, max_action)
    e_time = time.time()
    print("TTI running time is:", e_time - s_time)
# with h5py.File("../plot_data/pf_data_16_4_16_mobile.hdf5", "w") as data_file:
#         # data_file.create_dataset("snr", data=pf_ur_min_snr)
#     data_file.create_dataset("history", data=pf_ue_history_steps)

# with h5py.File("../plot_data/mt_data_16_4_16_mobile.hdf5", "w") as data_file:
#         # data_file.create_dataset("snr", data=ur_min_snr)
#     data_file.create_dataset("history", data=max_ue_history_steps)

# with h5py.File("../plot_data/pre_processing_16_4_16_mobile.hdf5", "w") as data_file:
#     data_file.create_dataset("pre", data=pre_data)

print('Processing is finished\n')



end_time = time.time()

running_time = (end_time - start_time)

print('Running Time is:', running_time)
