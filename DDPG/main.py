import os
import numpy as np
import logging
from train_test import train, test
import warnings
import time
from arg_parser import init_parser
# from setproctitle import setproctitle as ptitle
from mimo_sim_ul import *
import h5py

if __name__ == "__main__": ### set at as main
    # ptitle('WOLP_DDPG') ## changing python process name
    warnings.filterwarnings('ignore')
    parser = init_parser('WOLP_DDPG')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)[1:-1] ### e.g. gpu_ids = [0,1], [1:-1]: '0,1' (remove '['and']'])

    from util import get_output_folder, setup_logger
    from wolp_agent import WolpertingerAgent

    args.save_model_dir = get_output_folder('../output','QuaDriGa')

    ### Import data from hdf5 file #############################

    # Import se_max and H

    data_shuffled_file  = h5py.File('../../SAC_KNN/16_4_16_clu4_all.hdf5', 'r')
    se_max_ur = data_shuffled_file.get('se_max')
    se_max_ur = np.array(se_max_ur)
    H_r = np.array(data_shuffled_file.get('H_r'))
    H_i = np.array(data_shuffled_file.get('H_i'))
    # H_i = np.transpose(H_i,(2,1,0))
    # H_r = np.transpose(H_r,(2,1,0))
    print("H_r shape is:", H_r.shape)
    print("H_i shape is:", H_i.shape)
    H = np.array(H_r + 1j*H_i)
    print("H shape is:", H.shape)
    print("se_max shape is:", se_max_ur.shape)

    N_file = h5py.File('../../SAC_KNN/16_4_16_clu4_normal.hdf5','r')
    normal_ur = np.array(N_file.get('normal'))
    print("normal_ur shape is:", normal_ur.shape)
    
    pre_file = h5py.File('../../SAC_KNN/plot_data/pre_processing_16_4_16_clu4.hdf5','r')
    pre_data = np.array(pre_file.get('pre'))
    print("pre_data shape is:", pre_data.shape)
    #############################################################

    ### Basic parameters
    nb_states = 544
    nb_actions = 1  # the dimension of actions, usually it is 1. Depend on the environment.
    max_actions = 2516 #(4UEs and 16QAM)
    continuous = False


    if args.seed > 0:
        np.random.seed(args.seed)
        # env.seed(args.seed)

    if continuous:
        agent_args = {
            'continuous': continuous,
            'max_actions': None,
            'action_low': action_low,
            'action_high': action_high,
            'nb_states': nb_states,
            'nb_actions': nb_actions,
            'args': args,
        }
        print('Action is Continuous')
    else:
        agent_args = {
            'continuous': continuous,
            'max_actions': max_actions,
            'action_low': None,
            'action_high': None,
            'nb_states': nb_states,
            'nb_actions': nb_actions,
            'args': args,
        }
        print('Action is Discrete')
    
    agent = WolpertingerAgent(**agent_args)

    if args.load:
        agent.load_weights(args.load_model_dir)

    if args.gpu_ids[0] >= 0 and args.gpu_nums > 0:
        agent.cuda_convert()


    # set logger, log args here
    log = {}
    if args.mode == 'train':
        setup_logger('RS_log', r'{}/RS_train_log'.format(args.save_model_dir))
    elif args.mode == 'test':
        setup_logger('RS_log', r'{}/RS_test_log'.format(args.save_model_dir))
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
    log['RS_log'] = logging.getLogger('RS_log')
    d_args = vars(args)
    d_args['max_actions'] = args.max_actions
    for key in agent_args.keys():
        if key == 'args':
            continue
        d_args[key] = agent_args[key]
    for k in d_args.keys():
        log['RS_log'].info('{0}: {1}'.format(k, d_args[k]))

    start_time = time.time()
    
    if args.mode == 'train':

        train_args = {
            'continuous': continuous,
            # 'env': env,
            'agent': agent,
            'max_episode': args.max_episode,
            'warmup': args.warmup,
            'save_model_dir': args.save_model_dir,
            'max_episode_length': args.max_episode_length,
            'logger': log['RS_log'],
            'save_per_epochs': args.save_per_epochs,
            'H_r': H_r,
            'H_i': H_i,
            'normal_ur': normal_ur,
            'pre_data': pre_data,
            'se_max_ur': se_max_ur,
            'epsilon': args.epsilon
        }
        print('Starting Training')

        train(**train_args)
        end_time = time.time()
        print('Training time is:', (end_time-start_time))
    elif args.mode == 'test':

        test_args = {
            # 'env': env,
            'agent': agent,
            'model_path': args.load_model_dir,
            'test_episode': args.test_episode,
            'max_episode_length': args.max_episode_length,
            'logger': log['RS_log'],
            # 'save_per_epochs': args.save_per_epochs,
            'H': H,
            'se_max_ur': se_max_ur

        }

        test(**test_args)
        end_time = time.time()
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
