#!/usr/bin/env python
# coding: utf-8

# In[1]:


from threading import Thread
import numpy as np
from Config import Config


# In[2]:


class ThreadTrainer(Thread):
    def __init__(self, server, id):
        super(ThreadTrainer, self).__init__()
        self.setDaemon(True)
        
        self.id = id
        self.sever = server
        self.exit_flag = False
    
    def run(self):
        while not self.exit_flag:
            batch_size = 0
            while batch_size <= Config.TRAINING_MIN_BATCH_SIZE:
                x_, r_, a_ = self.server.training_q.get()
                if batch_size == 0:
                    x__ = x_; r__ = r_; a__ = a_
                else:
                    x__ = np.concatenate((x__, x_))
                    r__ = np.concatenate((r__, r_))
                    a__ = np.concatenate((a__, a_))
                batch_size+=x_.shape[0]
                
            if Config.TRAIN_MODELS:
                self.server.train_model(x__, r__, a__, self.id)


# In[ ]:




