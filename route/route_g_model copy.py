import xlwt
import torch
import os
import numpy as np
from pathlib import Path
import sys
import random

sys.path.append("..")
from algorithms.attention_pretrain import AttentionPretrain 
from algorithms.attention_sac import AttentionSAC
from algorithms.attention_ddpg import AttentionDDPG
from env.UAVEnv import UAVnet
from utils.arguments import parse_args
from route_map import test

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    
    config = parse_args()
    # config.reward_form='LT'
    setup_seed(config.seed)
    # config.obs_radius = 250
    uav_env = UAVnet(arglist=config)
    wb = xlwt.Workbook()
    # 添加一个表
    ws = wb.add_sheet('test')

    model_index = 40
    run_index = 201
    model_type = 'pre_actor'
    model_dir = Path('../models') / model_type / ('run%i' % run_index) / ('model.pt')
    # model_dir = Path('../models') / model_type / ('run%i' % run_index) / 'incremental'/('model_%i.pt'% model_index)

    model = AttentionDDPG.init_from_save(config, model_dir, load_critic=False)

    share_obs,obs = uav_env.reset()
    model.prep_training(device='gpu')
    # model.prep_training(device='gpu')
    for i in range(config.n_step):
        # print(i)
        obs = np.array(obs)
        obs = torch.FloatTensor(obs).cuda()
        share_obs = np.array(share_obs)
        share_obs = torch.FloatTensor(share_obs).cuda()
        attention_state ,postion_encode= model.attention_net.forward(share_obs,obs)
        actions, log_pi, = model.policy.forward(attention_state, postion_encode)
        # print("action and std")
        # print(actions)
        # print(log_pi)
        # actions=actions.cpu().detach().numpy()
        # q1,q2=model.critic(attention_state,postion_encode,actions)
        # qmin=min(q1[0].item(),q2[0].item())
        # print(qmin)
        
        actions = torch.FloatTensor([[ac[0][0], ac[0][1]] for ac in actions])
        next_share_obs,next_obs, rewards, dones, infos = uav_env.step(actions)
         
        # print(actions)
        # print(dones)

        obs = next_obs
        share_obs=next_share_obs 
        # print("action: {}".format(actions))
        # print("reward: {}".format(rewards.flatten()))
        uav_env.render(ws)
    print("over")
    model_name=('%s_%i_model%i' % (model_type, run_index, model_index))
    excel_path = Path('../excel') / ('%s.xls' % model_name)
    wb.save(excel_path)
    test(model_name)