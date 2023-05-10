from itertools import combinations
from itertools import product
import numpy as np 
from mimo_sim_ul import *
import torch
import h5py

def train(continuous, agent, max_episode, warmup, save_model_dir, max_episode_length, logger, save_per_epochs, H_r, H_i, normal_ur,pre_data, se_max_ur,epsilon):
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    s_t = None
    user_set = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    # mod_set = [4]
    a = 1
    b = 1
    c = 0
    random = 1
    reward_scale = 0.1
    reward_record = np.zeros((1000,))
    history_record = np.zeros((10,1000,16))
    ue_history = 0.01*np.ones((16,)) ## Tp history for each UE [4,]
    H = np.array(H_r + 1j*H_i)
    def sel_ue(action):
        sum_before = 0
        for i in range (1,5):
            sum_before += len(list(combinations(user_set, i)))
            if ((action+1)>sum_before):
                continue
            else:
                idx = i
                sum_before -= len(list(combinations(user_set, i)))
                ue_select = list(combinations(user_set, i))[action-sum_before]
                break
        return ue_select,idx

    # def sel_mod (action,idx,sum_before):
    #     ue_remain = action + 1 - sum_before
    #     num_1 = ue_remain // (3**idx) ## ue order
    #     ue_select = list(combinations(user_set, idx))
    #     ue_select = np.array(ue_select[num_1])
    #     mod_remain = ue_remain - num_1*(3**idx)
    #     mod_select = list(product(mod_set, repeat = idx))
    #     mod_select = np.array(mod_select[mod_remain-1])
    #     return ue_select,mod_select

    
    while episode < max_episode:
        
        print('Training processing: ',episode,' episode\n')
        while True:
            # if episode_steps == 398:
                # print('Training processing: ',episode,' episode ',episode_steps,' episode_steps')
            if s_t is None:
                s_t = np.concatenate((np.reshape(se_max_ur[0,:],(1,16)),np.reshape(ue_history,(1,16)),np.reshape(H_r[0,:,:],(1,-1)),np.reshape(H_i[0,:,:],(1,-1))),axis = 1)
                # print('s_t shape and type is',s_t.shape)
                agent.reset(s_t)

            # agent pick action ...
            # args.warmup: time without training but only filling the memory
            if step <= warmup or (step > warmup and random > np.random.rand(1)):
            # if step <= warmup or (episode < 100):
                action_array = agent.random_action()
                action = action_array[0]
                
                print('Random action')
            # if step <= warmup:
            #     action_array = agent.random_action()
            #     action = action_array[0]
                # print('Random action')
            else:
                action_array = agent.select_action(s_t)
                action = action_array[0]
                
                print('Actor action')
            
            random -= 1/epsilon
            # print('random is:', random)
            # print('action is',action,'action type is',type(action))
            
            ue_select,idx = sel_ue(action)
            
            print('ue_selection is:', ue_select)
            ur_se_total = pre_data[episode_steps,action,0]
            ur_min_snr = pre_data[episode_steps,action,1]
            ur_se = pre_data[episode_steps,action,2:2+idx]
            # mod_select = np.ones((idx,)) *16 # 16-QAM
            # ur_se_total, ur_min_snr, ur_se = data_process(np.reshape(H[episode_steps,:,ue_select],(16,-1)),idx,mod_select)
            ur_se_total = ur_se_total/normal_ur[episode_steps]
            
              
            print('se_total, min_snr are',ur_se_total, ur_min_snr)
            print('idx is',idx)
            for i in range(0,idx):
                ue_history[ue_select[i]] += ur_se[i]
                # print('ur_se is',ur_se[i])
            
            if (episode >= 990):
                history_record[episode-990,episode_steps,:] = ue_history
            # if episode_steps == 398:
            #     print('ue_history is:',ue_history)
            jfi = np.square((np.sum(ue_history))) / (16 * np.sum(np.square(ue_history)))
            
            print('jfi is', jfi)
            # s_t1 = [se_max_ur[(episode_steps+1),:],ue_history,H[(episode_steps+1),:,:]]
            s_t1 = np.concatenate((np.reshape(se_max_ur[(episode_steps+1),:],(1,16)),np.reshape(ue_history,(1,16)),np.reshape(H_r[(episode_steps+1),:,:],(1,-1)),np.reshape(H_i[(episode_steps+1),:,:],(1,-1))),axis = 1)
            r_t  = (a*ur_se_total + b*jfi + c*ur_min_snr)*reward_scale
            # r_t  = ur_se_total*jfi*ur_min_snr *0.001
               
            print('reward is:', r_t)
            done = False

            if max_episode_length and episode_steps >= max_episode_length - 2:
                done = True

            # agent observe and update policy
            agent.observe(r_t, s_t1, done) ## Store reply buffer and update state to next state
            if step > warmup:
                agent.update_policy()

            # update
            step += 1
            episode_steps += 1
            episode_reward += r_t
            print('episode reward is:', episode_reward,'\n')
            s_t = s_t1
            # s_t = deepcopy(s_t1)

            if done:  # end of an episode
                logger.info(
                    "Ep:{0} | R:{1:.4f}".format(episode, episode_reward)
                )
                reward_record[episode] = episode_reward
                agent.memory.append(
                    s_t,
                    agent.select_action(s_t),
                    0., True
                )

                # reset
                s_t = None
                episode_steps =  0
                episode_reward = 0.
                episode += 1
                ue_history = 0.01*np.ones((16,)) ## Tp history for each UE [4,]
                # break to next episode
                break
        # [optional] save intermideate model every run through of 32 episodes
        if step > warmup and episode > 0 and episode % save_per_epochs == 0:
            agent.save_model(save_model_dir)
            logger.info("### Model Saved before Ep:{0} ###".format(episode))

        # if(episode == 800):
        #     with h5py.File("../ddpg_data/plotdata_half_shuffled.hdf5", "w") as data_file:
        #         data_file.create_dataset("reward", data=reward_record)
        #         data_file.create_dataset("history", data=history_record)
        #     print('Half Training is finished\n')


    with h5py.File("./ddpg_16416_1_1_mob.hdf5", "w") as data_file:
        data_file.create_dataset("reward", data=reward_record)
        data_file.create_dataset("history", data=history_record)
    print('Training is finished\n')


def test(agent, model_path, test_episode, max_episode_length, logger, H, se_max_ur):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()

    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    episode_steps = 0
    episode_reward = 0.
    s_t = None
    min_snr_sum = 0
    ue_history = 0.01*np.ones((16,)) ## Tp history for each UE [4,]
    history_record = np.zeros((1000,16))
    reward_record = np.zeros((200,))

    user_set = [0,1,2,3]
    # mod_set = [4,16,64]
    a = 0.07
    b = 0.5
    c = 0.1

    def sel_ue(action):
        sum_before = 0
        for i in range (1,5):
            sum_before += len(list(combinations(user_set, i)))
            if ((action+1)>sum_before):
                continue
            else:
                idx = i
                sum_before -= len(list(combinations(user_set, i)))
                ue_select = list(combinations(user_set, i))[action-sum_before]
                break
        return ue_select,idx


    for t in range(test_episode):
        while True:
            if s_t is None:
                s_t = np.concatenate((np.reshape(se_max_ur[2000,:],(1,16)),np.reshape(ue_history,(1,16)),np.reshape(H[2000,:,:],(1,-1))),axis = 1)
                # print('s_t shape and type is',s_t.shape)
                agent.reset(s_t)


            action_array = policy(s_t)
            action = action_array[0]


            ue_select,idx = sel_ue(action)
            print('ue_selection is:', ue_select)
            mod_set = np.ones((idx,)) *16 # 16-QAM
            ur_se_total, ur_min_snr, ur_se = data_process(np.reshape(H[2000+episode_steps,:,ue_select],(8,-1)),idx,mod_set)

            min_snr_sum += ur_min_snr

            for i in range(0,idx):
                ue_history[ue_select[i]] += ur_se[i]

            jfi = np.square((np.sum(ue_history))) / (4 * np.sum(np.square(ue_history)))
            print('jfi is', jfi)
            s_t1 = np.concatenate((np.reshape(se_max_ur[(2000+episode_steps+1),:],(1,4)),np.reshape(ue_history,(1,4)),np.reshape(H[(2000+episode_steps+1),:,:],(1,-1))),axis = 1)
            r_t  = a*ur_se_total + b*jfi + c*ur_min_snr
            print('reward is', r_t)
            done = False
            if (t == 120):
                history_record[episode_steps,:] = ue_history
            episode_steps += 1
            episode_reward += r_t
            
            if max_episode_length and episode_steps >= max_episode_length - 1:
                done = True
            if done:  # end of an episode
                print('t = :',t)
                logger.info(
                    "Ep:{0} | R:{1:.4f} SNR:{2:.4f}".format(t+1, episode_reward, min_snr_sum/max_episode_length)
                )
                s_t = None
                ue_history = 0.01*np.ones((4,)) ## Tp history for each UE [4,]
                reward_record[t] = episode_reward
                episode_reward = 0
                episode_steps = 0
                break

        with h5py.File("../plotdata_0501_ue_history.hdf5", "w") as data_file:
            data_file.create_dataset("history", data=history_record)
            data_file.create_dataset("reward", data=reward_record)