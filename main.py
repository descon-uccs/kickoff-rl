# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 15:28:19 2025

@author: Philip Brown
"""

from  environments import GridWorld
from util import record_video
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.evaluation import evaluate_policy

if __name__=="__main__" :
    
    gym.register('GridWorld',GridWorld)
        
    # uglyVideo(10,env)
    
    
    # vec_env = make_vec_env(InertialContinuousArena, n_envs=1, env_kwargs=dict(arena_size=10))
    
    env = GridWorld(5)
    
    model = DQN("MlpPolicy", env, verbose=1, device='cpu')
    
    record_video('GridWorld', model,prefix='untrained')
    
    # Train the agent
    model.learn(total_timesteps=10_000)
    record_video('GridWorld', model,prefix='10ksteps')