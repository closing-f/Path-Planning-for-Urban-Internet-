import torch
import os
import numpy as np
from pathlib import Path
from tensorboardX import SummaryWriter
import sys

sys.path.append("..")
from utils.buffer import ReplayBuffer
from algorithms.attention_pretrain import AttentionPretrain
import time
from env.UAVEnv import UAVnet
from utils.arguments import parse_args


def run_from_save(config, run_dir, s_index, current_dir_num,filename):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    env = UAVnet(config)
    if current_dir_num == s_index:
        model = AttentionPretrain(config)
    
    else:
        model_dir =('run%i' % (current_dir_num-1)) / 'model.pt'
        model = AttentionPretrain.init_from_save(model_dir)
    curr_run = 'run%i' % current_dir_num
    curr_run = run_dir / curr_run

    replay_buffer = ReplayBuffer(config)
    replay_buffer.load(filename)
    
    ep_i = 0
    save_num=1
    while (ep_i < 60000):
        ep_i+=1
        model.prep_training(device='gpu')
        loss=0
        for u_i in range(50):
            sample = replay_buffer.sample(1,
                                            to_gpu=config.use_gpu)
            share_state_batch, state_batch, action_batch, log_pi_batch, reward_batch, next_share_state_batch, next_state_batch, dones = sample
            
            print(share_state_batch)
            # print(state_batch)
            # print(action_batch)
            print(next_share_state_batch)
            # print(next_state_batch)
            # print(dones)


            # model.test_parameters()
        print('loss: ',end='')
        print(loss)
        model.update_all_targets()

        if ep_i % 50 == 0:
            model.prep_rollouts(device='cpu')

            os.makedirs(curr_run / 'incremental', exist_ok=True)
            model.save(curr_run / 'incremental' / ('model_%i.pt'% save_num))
            model.save(curr_run / 'model.pt')
            print("save over")
            save_num+=1
            

    env.close()


if __name__ == '__main__': 
    
    modeldir = Path('./models')
    config = parse_args()
    # config.attention_use_cnn=True
    # config.reward_form='LT'
    run_dir = modeldir / 'pre_actor'
    s_index = 13
    c_index = 13
    run_from_save(config, run_dir, s_index, c_index,'../data/expand_env/greedy_IA_xy.json')
'''run6 use_cnn lt
   
   run7 lt
   run8  xy ia 512 900+100 plus
   run11 e
   run12 xy ia 512 900+100 
    '''