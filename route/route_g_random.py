import xlwt
import torch
import os
import numpy as np
from pathlib import Path
import sys 
from route_map import test
sys.path.append("..") 
from env.UAVEnv import UAVnet
from utils.arguments import parse_args
import copy

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    config = parse_args()
    # config.reward_form='RE'
    config.obs_radius = 300
    uav_env = UAVnet(config)
    
    wb = xlwt.Workbook()
    # 添加一个表
    ws = wb.add_sheet('test')
    ws2=wb.add_sheet('ctime')
    model_index = 3
    run_index = 1
    model_type = 'random'
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    nr_pur = uav_env.nr_agents
    # torch.manual_seed(config.seed)
    # np.random.seed(config.seed)
    share_obs, obs = uav_env.reset()
    
    for i in range(config.n_step):
        # print(i)
        obs = np.array(obs)
        obs = torch.FloatTensor(obs)
        share_obs = np.array(share_obs)
        share_obs = torch.FloatTensor(share_obs)
        c_time = np.zeros(100)
        c_time_square = np.zeros(100)
        ac = np.ones((nr_pur, 2)) * 0.5
        nodes = uav_env.world.agents
        evaders = nodes[-uav_env.nr_evaders:]
        pursuers = nodes[:-uav_env.nr_evaders]

        for k in range(nr_pur):
            # 从算过的dis matrix里找到对应的PoI的距离，与角度
            evader_dists = uav_env.world.distance_matrix[k, :][-100:]
            # evader_bearings = uav_env.world.angle_matrix[k, :][-100:]
            evader_deltax=uav_env.world.delta_matrix[k]
            evader_deltax=evader_deltax[-100:,:]
            # print(evader_deltax.shape)
            
            
            # 排序
            dist_argsort = np.argsort(evader_dists)
            # 找没有被覆盖的PoIs
            for arg in dist_argsort:
                if evaders[arg].cover == 0:
                    ac[k,0]=0.8* evader_deltax[arg][0]/evader_dists[arg]
                    ac[k,1]=0.8* evader_deltax[arg][1]/evader_dists[arg]
                    # ac[k, 1] = -evader_bearings[arg]
                    break
        actions = torch.FloatTensor([[a[0],a[1]] for a in ac])
        print(actions)
        # print("```")
        next_share_obs,next_obs, rewards, dones, infos = uav_env.step(actions)
        # print(i,end='')
        # print(dones)
        # print(rewards)
        obs = next_obs
        share_obs=next_share_obs
        for j in range(uav_env.nr_evaders):
            # c_time[i] = copy.deepcopy(evaders[i].c_time) for i in range(uav_env.nr_evaders):
            c_time[j] = copy.deepcopy(evaders[j].c_time)
            c_time_square[j] = np.square(c_time[j] / config.n_step)

        c_time = c_time / config.n_step
        SumCTime = sum(c_time)
        SumCTimeSquare = sum(c_time_square)
        # 这里的100是PoI的个数
        AverageCoverageScore = SumCTime / 100
        # print(uav_env.nr_evaders)

        for j in range(uav_env.nr_evaders):
            ws2.write(i, j, c_time[j])
        # print("action: {}".format(actions))
        # print("reward: {}".format(rewards.flatten()))
        uav_env.render(ws)
    model_name=('%s_%i_model_%i' % (model_type, run_index, model_index))
    excel_path = Path('../excel') / ('%s.xls' % model_name)
    wb.save(excel_path)
    test(model_name)
    