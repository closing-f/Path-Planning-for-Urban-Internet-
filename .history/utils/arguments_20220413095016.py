import time
import torch
from torch import nn
import argparse
from typing import Dict, List, Tuple, Type, Union

device =  torch.device('cpu')
# print(device)
time_now = time.strftime('%y%m_%d%H%M')


def parse_args():
    time_now = time.strftime('%y%m_%d%H%M')
    parser = argparse.ArgumentParser("PS-based RL experiments for multi-UAV networks")
    parser.add_argument("--nb_cargos", default=15, type=int)

    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_episodes", default=9000000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    # parser.add_argument("--buffer_length", default=4096, type=int)
    # parser.add_argument("--steps_per_update", default=16, type=int)
    # parser.add_argument("--num_updates", default=100, type=int, )
    # parser.add_argument("--batch_size", default=16, type=int)
    # parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--buffer_length", default=10240, type=int)
    parser.add_argument("--steps_per_update", default=10240, type=int)
    parser.add_argument("--num_updates", default=15, type=int,)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--save_interval", default=5, type=int)

    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.00001, type=float)
    parser.add_argument("--q_lr", default=0.00001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", default=True, action='store_true')
    parser.add_argument("--run_index", default=1, type=int)
    parser.add_argument("--attention_use_cnn", default=False, type=bool)
    parser.add_argument("--current_index", default=1, type=int)

    # 以下为PStorch参数
    # Model save and load parameters
    parser.add_argument("--Model", type=str, default='SAC', help="PPO DDPG DDPG Actor Critic")
    parser.add_argument("--seed", type=int, default=123, help="Random Seed")
    parser.add_argument("--reward_form", type=str, default='LT', help="LT PC IA RE AL")
   

    # inequality
    parser.add_argument("--beta", type=float, default=0.05, help="base is 0.05")
    parser.add_argument("--alpha", type=float, default=5, help="base is 5")
    parser.add_argument('--alphaT', type=float, default=0.5, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')

    # UAV_environment
    parser.add_argument("--time_dim", type=int, default=9, help="the number of UAV agents")
    parser.add_argument("--device", type=str, default=device)
    parser.add_argument("--start_time", type=str, default=time_now, help="the time when start the game")
    parser.add_argument("--nb_UAVs", type=int, default=6, help="the number of UAV agents")
    parser.add_argument("--max_energy", type=int, default=4, help="max energy")
    parser.add_argument("--max_weight", type=int, default=4, help="max energy")
    parser.add_argument("--comm_radius", type=float, default=500, help="communication range")
    parser.add_argument("--obs_radius", type=float, default=250, help="coverage range")
    # Fixed UAV parameters
    parser.add_argument("--torus", type=bool, default=False, help="If True, the world is cycle; Fixed para")
    parser.add_argument("--uav_obs_dim", type=int, default=4,
                        help="x, y, energy, pursuer(bool), all(bool); Fixed para")
    parser.add_argument("--local_obs", type=int, default=11,
                        help="time encoding(0~256) 9 bit + if_wall")
    parser.add_argument("--nb_PoIs", type=int, default=100, help="the number of PoIs; Fixed para")
    parser.add_argument("--cargo_dim", type=int, default=4, help="deltax,deltay, cover(bool), c_time; Fixed para")
    parser.add_argument("--action_dim", type=int, default=1, help="cargo index")
    parser.add_argument("--n_step", type=int, default=256, help="the time of mission time; Fixed para")
    parser.add_argument("--max_step", default=2, help="step in max_step")
    parser.add_argument("--world_size", type=int, default=100, help="The world is Square grid; Fixed para")
    # base.py -> action_repeat = 10; self.dt = 0.01; self.max_lin_v = 50; self.max_ang_v = np.pi
    # base.py -> energy_fun = (1/3.0*v**3 - 0.0625*v + 0.03)/30=0.002
    # UAV.py -> self.energy = 3; self.dim_rec_o = (8, 6); self.dim_evader_o = (self.n_evaders, 5);
    # UAV.py -> self.dim_local_o = 2
    # Other UAV parameters
    parser.add_argument("--obs_mode", type=str, default='sum_obs_no_ori')
    parser.add_argument("--dynamics", type=str, default='unicycle')
    parser.add_argument("--distance_bins", type=int, default=8)
    parser.add_argument("--bearing_bins", type=int, default=8)

   
    parser.add_argument("--activation_fn", type=Type[nn.Module],
                        default=nn.ReLU, help="in SB3, default is nn.Tanh")

    # Train steps
    parser.add_argument("--train_steps", type=int, default=10240,
                        help="the train_steps of once model update， UAV is 1024 * 10")
    parser.add_argument("--train_episodes", type=int, default=40, help="Train episodes / 10")

    # evaluation parameters
    parser.add_argument("--episodes", type=int, default=3, help="the number of eval_episodes")
    parser.add_argument("--test_model_path_file", type=str, default="C:/Users/user/Desktop/env/save/PPO/20201214",
                        help="model load path")

    return parser.parse_args()
