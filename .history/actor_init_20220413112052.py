import torch
import os
import numpy as np
from pathlib import Path
from tensorboardX import SummaryWriter

from algorithms.attention_pretrain import AttentionPretrain
import time
from env.UAVEnv import UAVnet
from utils.arguments import parse_args


def run_from_save(config, run_dir, s_index, current_dir_num,filename):
    # torch.manual_seed(config.seed)
    # np.random.seed(config.seed)
    if  s_index==1:
        # model = AttentionPretrain(config)
        
        model = AttentionPretrain(config)
    else:
        model_dir =('run%i' % (current_dir_num)) / 'model.pt'
        model = AttentionPretrain.init_from_save(model_dir)
    curr_run = 'run%i' % current_dir_num
    curr_run = run_dir / curr_run

    
    # replay_buffer.load(filename)
    uav_env = UAVnet(config)
    ep_i = 0
    save_num=s_index
    model.prep_rollouts(device='gpu')
    while (ep_i < 6000):
        obs = uav_env.reset()
        buffer_index=0
        while buffer_index < 2048:   
            obs = np.array(obs)  
            
            obs = torch.Tensor(obs).cuda()
            action=model.step(obs)
            # print(action.shape)

            actions = torch.Tensor([[ac[0][0]] for ac in action])
            next_obs, rewards, dones, infos = uav_env.step(actions)
                # share_obs,observations, actions, log_action, value_pred,rewards, next_share_obs,next_observations, dones
             
            # print(dones)
            if dones[0]:
                obs=uav_env.reset()
                # print("reset")
            else:
                obs = next_obs
                
    
        ep_i+=1
        # print(ep_i)
        model.prep_training(device='gpu')
   
        for u_i in range(5):
            # print("update")
            sample = replay_buffer.preactor_sample(config.batch_size,
                                            to_gpu=config.use_gpu)
            model.pretrain_actor(sample)
            # model.test_parameters(
        
        
        if ep_i % 5 == 0:
            model.prep_rollouts(device='cpu')

            os.makedirs(curr_run / 'incremental', exist_ok=True)
            model.save(curr_run / 'incremental' / ('model_%i.pt'% save_num))
            model.save(curr_run / 'model.pt')
            print("save over")
            save_num+=1



if __name__ == '__main__': 
    
    modeldir = Path('./models')
    config = parse_args()
    # config.attention_use_cnn=True
    # config.reward_form='LT'
    run_dir = modeldir / 'pre_actor'
    s_index = 1
    c_index = 1
    run_from_save(config, run_dir, s_index, c_index,'./data/TRPO/greedy_LT_8.json')
'''
run3 完全随机位置 219428
run2 位置限制0-100 
run4 位置限制0-100 219583
run5 成团0-100 + 900 
run100

'''