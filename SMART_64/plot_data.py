import numpy as np 
import time
import matplotlib.pyplot as plt 
import h5py


########################################
############Parameter Setting###########
########################################
ep = 0
pf_snr = np.zeros(400)
pf_ue_history = np.zeros((400,4))

sac_reward = np.zeros(800)
sac_ue_history = np.zeros((400,4))

sac_mod_reward = np.zeros(800)
sac_mod_ue_history = np.zeros((400,4))

sac_ug_reward = np.zeros(800)
sac_ug_ue_history = np.zeros((400,4))


########################################
############Read PF Data################
########################################

pf_file  = h5py.File('./plot_data/pf_data.hdf5', 'r')
pf_snr = np.array(pf_file.get('snr'))
pf_ue_history = np.array(pf_file.get('history'))

pf_se_total = np.sum(pf_ue_history,axis = 1)


pf_jfi = np.zeros(400)
for i in range (0,399):
    pf_jfi[i] = np.square((np.sum(pf_ue_history[i,:]))) / (4 * np.sum(np.square(pf_ue_history[i,:])))
    


########################################
############Read RR Data################
########################################

rr_file  = h5py.File('./plot_data/RR_data.hdf5', 'r')
rr_ue_history = np.array(rr_file.get('history'))

rr_se_total = np.sum(rr_ue_history,axis = 1)


rr_jfi = np.zeros(400)
for i in range (0,399):
    rr_jfi[i] = np.square((np.sum(rr_ue_history[i,:]))) / (4 * np.sum(np.square(rr_ue_history[i,:])))




########################################
############Read SAC Data###############
########################################

sac_file  = h5py.File('./plot_data/plotdata_shuffled_3.hdf5', 'r')

sac_reward = np.array(sac_file.get('reward'))

sac_ue_history = np.array(sac_file.get('history'))

sac_se_total = np.sum(sac_ue_history[ep,:,:],axis = 1)

sac_jfi = np.zeros(400)
for i in range (0,399):
    sac_jfi[i] = np.square((np.sum(sac_ue_history[ep,i,:]))) / (4 * np.sum(np.square(sac_ue_history[ep,i,:])))


########################################
#########Read SAC W/ MOD Data###########
########################################

sac_mod_file  = h5py.File('./plot_data/plotdata_shuffled_84_mod.hdf5', 'r')

sac_mod_reward = np.array(sac_mod_file.get('reward'))

sac_mod_ue_history = np.array(sac_mod_file.get('history'))

sac_mod_se_total = np.sum(sac_mod_ue_history[ep,:,:],axis = 1)

sac_mod_jfi = np.zeros(400)
for i in range (0,399):
    sac_mod_jfi[i] = np.square((np.sum(sac_mod_ue_history[ep,i,:]))) / (4 * np.sum(np.square(sac_mod_ue_history[ep,i,:])))


########################################
######Read SAC W/ UG & MOD Data#########
########################################

sac_ug_file  = h5py.File('./plot_data/plotdata_shuffled_ur_gp.hdf5', 'r')

sac_ug_reward = np.array(sac_ug_file.get('reward'))

sac_ug_ue_history = np.array(sac_ug_file.get('history'))

sac_ug_se_total = np.sum(sac_ug_ue_history[ep,:,:],axis = 1)

sac_ug_jfi = np.zeros(400)
for i in range (0,399):
    sac_ug_jfi[i] = np.square((np.sum(sac_ug_ue_history[ep,i,:]))) / (4 * np.sum(np.square(sac_ug_ue_history[ep,i,:])))




#########################################
##########Reward  vs. Epoches############
#########################################


plt.figure(figsize=(10, 8), dpi=80); 

plt.plot(sac_ug_reward, ms=1.0)
# plt.axis('square')
plt.xlim([0,799])
plt.grid()

plt.title('Reward vs. Epoch in Training Process',fontsize=18)
plt.xlabel('Traning Epoches',fontsize=18)
plt.ylabel('Reward',fontsize=18)
# plt.legend(["Rx","Tx"], loc = "upper right")
plt.show()


#########################################
##########Reward  vs. Epoches############
#########################################


plt.figure(figsize=(10, 8), dpi=80); 

plt.plot(sac_mod_reward, ms=1.0)
# plt.axis('square')
plt.xlim([0,799])
plt.grid()

plt.title('Reward vs. Epoch in Training Process',fontsize=20)
plt.xlabel('Traning Epoches',fontsize=18)
plt.ylabel('Reward',fontsize=18)
# plt.legend(["Rx","Tx"], loc = "upper right")
plt.show()

#########################################
###############SE  and  JFI##############
#########################################


plt.figure(figsize=(10, 8), dpi=80); 

# plt.subplot(221)
plt.plot(sac_se_total[0:399],'r',linewidth=1.5)
# plt.axis('square')
plt.xlim([0,399])
plt.grid()
plt.plot(sac_mod_se_total[0:399],'g',linewidth=1.5)
plt.plot(sac_ug_se_total[0:399],'c',linewidth=1.5)
plt.plot(pf_se_total[0:399],'b',linewidth=1.5)
plt.plot(rr_se_total[0:399],'y',linewidth=1.5)

plt.title('Total Spectral Efficiency Comparison',fontsize=20)
plt.xlabel('TTI',fontsize=18)
plt.ylabel('Total Spectral Efficiency (bits/Hz/TTI)',fontsize=18)
plt.legend(["SAC_KNN w/o Mod","SAC_KNN w/ Mod","SAC_KNN w/ UG+Mod","PF","RR"], loc = "lower right",fontsize=13)
plt.show()




plt.figure(figsize=(10, 8), dpi=80); 

# plt.subplot(221)
plt.plot(sac_jfi[0:399],'r',linewidth=1.5)
# plt.axis('square')
plt.xlim([0,399])
plt.grid()

plt.plot(sac_mod_jfi[0:399],'g',linewidth=1.5)
plt.plot(sac_ug_jfi[0:399],'c',linewidth=1.5)
plt.plot(pf_jfi[0:399],'b',linewidth=1.5)
plt.plot(pf_jfi[0:399],'y',linewidth=1.5)
plt.title('Fairness (JFI) Comparison',fontsize=20)
plt.xlabel('TTI',fontsize=18)
plt.ylabel('JFI',fontsize=18)
plt.legend(["SAC_KNN w/o Mod","SAC_KNN w/ Mod","SAC_KNN w/ UG+Mod","PF","RR"], loc = "lower right",fontsize=15)
plt.show()



####################################################
##############Condition Number plot#################
####################################################

# H_file = h5py.File('../1_out_3_all.hdf5','r')
# H_r = np.array(H_file.get('H_r'))
# H_i = np.array(H_file.get('H_i'))
# # H_i = np.transpose(H_i,(2,1,0))
# # H_r = np.transpose(H_r,(2,1,0))
# print("H_r shape is:", H_r.shape)
# print("H_i shape is:", H_i.shape)
# H = np.array(H_r + 1j*H_i)
# print("H shape is:", H.shape)

# cond_num = np.zeros(400)

# for i in range (0,400):
#     cond_num[i] = np.linalg.cond(H[i,:,:])

# plt.figure(figsize=(10, 8), dpi=80); 

# plt.plot(cond_num, ms=1.0)
# # plt.axis('square')
# plt.xlim([0,399])
# plt.grid()

# plt.title('Condition Number Of Channel Matrix',fontsize=20)
# plt.xlabel('TTI',fontsize=20)
# plt.ylabel('Condition Number',fontsize=20)
# # plt.legend(["Rx","Tx"], loc = "upper right")
# plt.show()