import torch
import os
import numpy as np
from pathlib import Path
import sys 
sys.path.append("..")
from utils.buffer import ReplayBuffer
from algorithms.attention_sac import AttentionSAC
import time
from env.UAVEnv import UAVnet
from utils.arguments import parse_args


def run_from_save(config, run_dir, start_index, current_dir_num, actor_model_index=1):
    env = UAVnet(config)
    model_index = 1
    # if current_dir_num == start_index:
    #     model_dir = Path('../models') / 'sac'/'run3'/'incremental'/ ('model_36.pt')
    # else:
    #     predir_num = current_dir_num - 1
    #     model_dir = run_dir / 'model_%i.pt'

    model = AttentionSAC(config)

    curr_run = run_dir / ('run%i' % current_dir_num)

    
    replay_buffer = ReplayBuffer(config)
    replay_buffer.load('../data/greedy_IA_test.json')
    ep_i = 0
    model.prep_rollouts(device='gpu')
    save_num = 1
    time_now = time.time()
    # 采集数据
    while ep_i < config.n_episodes:
        buffer_index = 0
        ep_i += 1
        while buffer_index < 2048:
            # 将环境重置，初始化，获得初始化observation
            model.prep_rollouts(device='gpu')
            
            obs = env.reset()
            eposide_index = 0
            # 这里只是因为 break 所以放缩了一下
            while eposide_index < 1025:
                eposide_index += 1
                obs = torch.Tensor(obs).cuda()
                # print(obs)
                action, log_action = model.step(obs)
                # print(action.shape) [6,1,2]

                # 运行轨迹时batch_size为1，
                action = torch.Tensor([[ac[0][0]*0.1, ac[0][1]*0.1] for ac in action])
                next_obs, rewards, dones, infos = env.step(action)
                
                replay_buffer.push(obs.cpu().detach().numpy(), action.cpu().detach().numpy(), rewards, next_obs, dones,log_action.cpu().detach().numpy())
                buffer_index += 1
                if dones[0]:
                    obs = env.reset()
                    print("Reset environment")
                    break
                else:
                    obs = next_obs

        if config.use_gpu:
            model.prep_training(device='gpu')
        else:
            model.prep_training(device='cpu')
        for u_i in range(10):
            sample = replay_buffer.sample(config.batch_size,
                                          to_gpu=config.use_gpu)
            model.update_critic(sample)
            model.update_policy(sample)
        model.update_all_targets()
        # todo prep_rollouts这个函数是要做什么


        # if ep_i % 20 == 0:
        #     model.prep_rollouts(device='cpu')
        #     os.makedirs(curr_run / 'incremental', exist_ok=True)
        #     model.save(curr_run / 'incremental' / ('model_%i.pt' % save_num))
        #     model.save(curr_run / 'model.pt')
        #     save_num+=1
    model.save(curr_run / 'model.pt')

    env.close()


if __name__ == '__main__':
    Run_dir = Path('./models') / 'test'
    s_index = 1
    c_index = 1
    Config=parse_args()
    run_from_save(Config, Run_dir, s_index, c_index)
