#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime
from multiprocessing import Process, Queue, Value


# In[2]:


import numpy as np
import time


# In[3]:


from Config import Config
from Environment import Environment
from Experience import Experience


# In[ ]:


class ProcessAgent(Process):
    def __init__(self, id, prediction_q, training_q, episode_log_q):
        super(ProcessAgent, self).__init__()
        
        self.id = id
        self.prediction_q = prediction_q
        self.training_q = training_q
        self.episode_log_q = episode_log_q
        
        self.env = Environment()
        self.num_actions = self.env.get_num_actions()
        self.actions = np.arange(self.num_actions)
        
        self.discount_factor = Config.DISCOUNT
        
        self.wait_q = Queue(maxsize=1)
        self.exit_flag = Value('i', 0)
        
    @staticmethod
    def _accumulate_reward(experiences, discount_factor, terminal_reward):
        reward_sum = terminal_reward
        for t in reversed(range(0, len(experiences)-1)):
            r = np.clip(experiences[t].reward, Config.REWARD_MIN, Config.RESULTS_MAX)
            reward_sum = discount_factor * reward_sum + r
            experiences[t].reward = reward_sum
        return experiences[:-1]
    
    def convert_data(self, experiences):
        x_ = np.array([exp.state for exp in experiences])
        a_ = np.eye(self.num_actions)[np.array([exp.action for exp in experiences])].astype(np.float32)
        r_ = np.array([exp.reward for exp in experiences])
        return x_, r_, a_
    
    def predict(self, state):
        self.prediction_q.put((self.id, state))
        p, v = self.wait_q.get()
        return p, v
    
    def select_action(self, prediction):
        if Config.PLAY_MODE:
            action = np.argmax(prediction)
        else:
            action = np.random.choice(self.actions, p = prediction)
        return action
    
    def run_episode(self):
        self.env.reset()
        done = False
        experiences = []
        
        time_count = 0
        reward_sum = 0.0
        
        while not done:
            if self.env.current_state is None:
                self.env.step(0)
                continue
                
            prediction, value = self.predict(self.env.current_state)
            action = self.select_action(prediction)
            reward, done = self.env.step(action)
            reward_sum += reward
            exp = Experience(self.env.previous_state, action, predictioni, reward, done)
            experiences.append(exp)
            
            if done or time_count == Config.TIME_MAX:
                terminal_reward = 0 if done else value
                
                updated_exps = ProcessAgent._accumulate_reward(experiences, self.discount_factor, terminal_reward)
                x_, r_, a_ = self.convert_data(updated_exps)
                yield x_, r_, a_, reward_sum
                
                time_count = 0
                
                experiences = [experiences[-1]]
                reward_sum = 0.0
                
            time_count += 1
            
    def run(self):
        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))
        
        while self.exit_flag.value == 0:
            total_reward = 0
            total_length = 0
            for x_, r_, a_, reward_sum in self.run_episode():
                total_reward += reward_sum
                total_length += len(r_) +1
                self.training_q.put((x_,r_,a_))
            self.episode_log_q.put((datetime.now(), total_reward, total_length))

