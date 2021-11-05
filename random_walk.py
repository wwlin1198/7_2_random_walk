import numpy as np
import random
import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np

'''
Homework 3: Exercise 7_2

Using N-step TD method for estimating V≈vpi

'''

class env:
    
    def __init__(self, num_states,policy):
        
        self.start_s = num_states//2 #start at mid
        self.curr_state = 0
        self.t_state = num_states
    
        self.reset()
        
    def reset(self):
        
        self.curr_state = self.start_s
         
    def step(self,action):
        
        state = self.curr_state
    
  
'''
Does the N-Step TD method for an alpha value
Inputs: policy, num_states

'''   
def n_stepTD(policy,state,alpha):
    
    _V = np.zeros(policy.shape)
    states = [state]
    rewards = [0]
    action_rand = list(range(9))
    
    time = float('inf')
    curr_time = 0
    while True:
        time += 1
        if curr_time < time:
            s = 3
            


          
def run_sim():
    
    NUM_STATES = 20
    EPISODES = 10
    RUNS = 100
    EPSILON = 0.1
    GAMMA = 1.0
    VARIANCE = 1
    
    policy = np.full((5, 5, 2),1/9) #initialize a policy 
    rw = env(NUM_STATES,policy)
    
        

    
            
        
        
        
        
        
        
if __name__ == "__main__":
    run_sim()