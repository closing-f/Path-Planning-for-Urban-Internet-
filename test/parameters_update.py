import xlwt
import torch
import os
import numpy as np
from pathlib import Path
import sys 
sys.path.append("..")
from algorithms.attention_ddpg import AttentionDDPG
from env.UAVEnv import UAVnet
from utils.arguments import parse_args


if __name__ == "__main__":

    model_index = 1
    config=parse_args()
    for i in range(1,6):
        print("model_index")
        print(model_index)
        model_dir = Path('../models') / 'pre_actor' / 'run11' / 'incremental' / ('model_%i.pt' % i)
        model = AttentionDDPG.test_parameter(config, model_dir,load_critic=False)
        model_index+=1

