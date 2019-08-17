#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys
if sys.version_info >= (3,0):
    from queue import Queue
else:
    from Queue import Queue

import numpy as np
import scipy.misc as misc

from Config import Config
from GameManager import GameManager


# In[6]:


class Environment:
    def __init__(self):
        self.game = GameManager(Config.ATARI_GAME, display=Config.PLAY_MODE)
        self.nb_frames = Config.STACKED_FRAMES
        self.frame_q = Queue(maxsize=self.nb_frames)
        self.previous_state = None
        self.current_state = None
        self.total_reward = 0
        
        self. reset()
        
    @staticmethod
    def _rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    
    @staticmethod
    def _preprocess(image):
        image = Environment._rgb2gray(image) # grayscale
        print(image)
        image = misc.imresize(image, [Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT], 'bilinear')
        print(image)
        image = image.astype(np.float32)/128.0-1.0
        print(image)
        return image
    
    def _get_current_state(self):
        if not self.frame_q.full():
            return None
        x_ = np.array(self.frame_q.queue)
        x_ = np.transpose(x_, [1,2,0])
        print(x_)
        return x_
    
    def _update_frame_q(self, frame):
        if self.frame_q.full():
            self.frame_q.get()
        image = Environment._preprocess(frame)
        self.frame_q.put(image)
        
    def step(self, action):
        observation, reward, done, _ = self.game.step(action)
        
        self.total_reward += reward
        self._update_frame_q(observation)
        
        self.previous_state = self.current_state
        self.current_state = self._get_current_state()
        return reward, done


# In[ ]:




