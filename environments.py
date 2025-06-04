# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 15:15:50 2025

@author: phili
"""
# new imports:
import gymnasium as gym
from gymnasium import spaces

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



class GridWorld(gym.Env) :
    
    metadata = {"render_modes": ['rgb_array']}
    
    LEFT = 0
    COAST = 1
    RIGHT = 2
    
    def __init__(self,  width=5, render_mode="rgb_array") :
        
        self.render_mode = render_mode
        
        self.width = width
        
        self.action_space = spaces.Discrete(3) # the three actions
        
        # equal to state space
        self.observation_space = spaces.Discrete(self.width)
        
        self.reset()
        
    def reset(self, seed=None, options=None) :
        '''
        This method sets self.state to the initial state and then return the current state
        '''
        self.state = self.width - 1 ## edit this line of code!
        return self.state, {}
    
    def step(self,action) :
        '''
        Input: action index
        Output: 
            - state after action
            - this stage reward
            - boolean: True if terminated, False if not terminated
        '''
        
        # state propagation
        motion = action - 1
        
        newState = max(0, 
                       min(self.width-1, self.state+motion))
        
        reward = 0
        if newState == 0 :
            reward += 10
        if action == self.LEFT or action== self.RIGHT :
            reward -= 1
        
        
        done = False
        if newState == 0 :
            done = True
            
        self.state = newState
        
        # returns observation, reward, terminated, truncated, info dict
        return self.state, reward, done, False, {}
    
    
    def render(self, mode='rgb_array'):
        assert mode == 'rgb_array', "Only 'rgb_array' mode is supported"
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(self.width, 1))
        canvas = FigureCanvas(fig)
    
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
        position = self.state
    
        # Draw background grid
        for i in range(self.width):
            rect = patches.Rectangle((i, 0), 1, 1, linewidth=1, edgecolor='gray', facecolor='white')
            ax.add_patch(rect)
    
        # Draw target at far left
        ax.add_patch(patches.Rectangle((0, 0), 1, 1, color='red', alpha=0.3))
    
        # Draw agent
        ax.add_patch(patches.Rectangle((position, 0), 1, 1, color='blue'))
    
    
        
        # Render to RGB array
        canvas.draw()
        buf = canvas.buffer_rgba()  # use RGBA buffer
        width, height = fig.get_size_inches() * fig.dpi
        image = np.frombuffer(buf, dtype=np.uint8).reshape(int(height), int(width), 4)[..., :3]  # strip alpha
        
        plt.close(fig)
    
        return image
    



class GridWorldInertial :
    
    # interpret these as "thrust" events:
    LEFT = 0
    COAST = 1
    RIGHT = 2
    
    def __init__(self,width) :
        self.reset()
        
    def reset(self) :
        '''
        This method sets self.state to the initial state and then return the current state
        '''
        
        # initialize
        
        self.state = {}
        
        # agent starts motionless at the far right:
        self.state['position'] = None
        self.state['velocity'] = None
        
        return self.state
    
    def step(self,action) :
        '''
        Input: action index
        Output: 
            - state after action
            - this stage reward
            - boolean: True if terminated, False if not terminated
        '''
        
        # interpretation: previous velocity propagates first, then is updated.
        
        
        # don't run off the right side of the world!
        
        # calculate rewards:
        reward = 0
        
        # check termination:
        done = False
            
        return self.state, reward, done
    
    def render(self, mode='rgb_array'):
        assert mode == 'rgb_array', "Only 'rgb_array' mode is supported"
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(self.width, 1))
        canvas = FigureCanvas(fig)
    
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
        position = self.state['position']
        velocity = self.state['velocity']
    
        # Draw background grid
        for i in range(self.width):
            rect = patches.Rectangle((i, 0), 1, 1, linewidth=1, edgecolor='gray', facecolor='white')
            ax.add_patch(rect)
    
        # Draw target at far left
        ax.add_patch(patches.Rectangle((0, 0), 1, 1, color='red', alpha=0.3))
    
        # Draw agent
        ax.add_patch(patches.Rectangle((position, 0), 1, 1, color='blue'))
    
        # Draw velocity arrow (rightward or leftward)
        if velocity != 0:
            direction = np.sign(velocity)
            arrow_length = min(abs(velocity), self.width - 1)
            arrow_x = position + 0.5
            arrow_dx = direction * arrow_length
            ax.arrow(
                arrow_x, 0.5,  # Start at center of agent cell
                dx=arrow_dx, dy=0,
                width=0.05, head_width=0.3, head_length=0.3,
                length_includes_head=True, color='black'
            )
    
        
        # Render to RGB array
        canvas.draw()
        buf = canvas.buffer_rgba()  # use RGBA buffer
        width, height = fig.get_size_inches() * fig.dpi
        image = np.frombuffer(buf, dtype=np.uint8).reshape(int(height), int(width), 4)[..., :3]  # strip alpha
        
        plt.close(fig)
    
        return image
    
    
if __name__ == '__main__' :
    
    gw = GridWorld(5)
    print(gw.reset())
    
    sample_actions = [0,0,1,2,0,0,0,0,1]
    cumulative_reward = 0
    for action in sample_actions :
        state,rew,done,_,_ = gw.step(action)
        print((state,rew,done))
        cumulative_reward += rew
        if done :
            break
    print(cumulative_reward)
    
    
# if __name__ == '__main__' :
    
#     gwi = GridWorldInertial(5)
#     print(gwi.reset())
    
#     sample_actions_gwi = [0,0,2,2,1,1,0,1,1,1,1]
#     for action in sample_actions_gwi :
#         print(gwi.step(action))