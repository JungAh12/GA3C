#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class Experience:
    def __init__(self, state, action, prediction, reward, done):
        self.state = state
        self.action = action
        self.prediction = prediction
        self.reward = reward
        self.done = done

