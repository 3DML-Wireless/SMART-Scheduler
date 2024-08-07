import sys
import numpy as np
import numpy.matlib
import time
import math
import os
import matplotlib
from itertools import combinations
import matplotlib.pyplot as plt
from mimo_sim_ul import *
import h5py


H_file  = h5py.File('./High_Mob_500_52_64_64.hdf5', 'r')


H_r = H_file.get('H_r')[:,:,20,:]
H_r = np.transpose(H_r,(2,1,0))
print("H_r shape is:", H_r.shape)

H_i = H_file.get('H_i')[:,:,20,:]
H_i = np.transpose(H_i,(2,1,0))
print("H_i shape is:",H_i.shape)

H = np.squeeze(np.array(H_r + 1j*H_i))
print("H shape is:",H.shape)


N_UE = 1
num_ue = 64
num_bs = 64

############ Normalization ##################
MOD_ORDER = np.array([16]) ## 

se_max_ur = np.zeros((500,num_ue)) 

for n_tti in range (0,500):
    H_T = H[n_tti,:,:]
    print('The number of round:',n_tti)
    # print(H_T.shape)
    for n_ur in range (0,num_ue):
        # print(H_T[:,n_ur].shape)
        H_matrix = np.reshape(H_T[:,n_ur],(num_bs,1))
        se_max_ur[n_tti,n_ur], _,_ = data_process(H_matrix,N_UE,MOD_ORDER)

normal_ur = np.zeros(500)
# se_max_ur = np.sort(se_max_ur,axis = -1)
for n_tti in range (0,500):
    normal_ur[n_tti] = np.sum(np.squeeze(se_max_ur[n_tti,:]))


with h5py.File("./High_Mob_pre_process.hdf5", "w") as data_file:
    data_file.create_dataset("se_max", data=se_max_ur)
    data_file.create_dataset("H_r", data=H_r)
    data_file.create_dataset("H_i", data=H_i)
    data_file.create_dataset("normal", data=normal_ur)
