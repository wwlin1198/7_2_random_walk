import numpy as np
import random
import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np
import itertools
from itertools import permutations

'''
Homework 3: Exercise 7_2

Author: Willy Lin

Using N-step TD method for estimating V≈vpi

'''

class env:
    
    def __init__(self, num_states):     
         
        self.start_s = num_states//2 #start at middle 
        self.curr_state = num_states//2 #same as the start state in the beginning
        self.terminal_state_1 = num_states  #the right most is the terminal state
        self.terminal_state_2 = 0 # the left most is also the terminal state
        self.num_states = num_states #total number of states
                
    def reset(self):
        self.curr_state = self.start_s
    
    #one step in simulation
    def step(self,action):
        reward = 0
        state = self.curr_state
        self.curr_state += action
        stop = False
        
        if self.curr_state >= self.terminal_state_1:
            reward = 1
            stop = True
        elif self.curr_state <= self.terminal_state_2:
            reward = -1
            stop = True
        else:
            reward = 0

        return stop, state, action, reward, self.curr_state
    
  
'''
Does the N-Step TD method for an alpha value using the random walk example from 6.2
Inputs: rw_env, ALPHAS, EPSILON, GAMMA, EPISODES

'''   
def n_stepTD(rw_env,ALPHAS,n,EPSILON=0.1,GAMMA=1.0,EPISODES=10):
    
    #defines the actions agent can take
    right = 1
    left = -1
    action_list = [right,left]
    
    #holds the actual values based on bellman
    policy =  np.zeros(rw_env.num_states)
    for i in range(1,rw_env.num_states+1):
        policy[i - 1] = i/(rw_env.num_states+1) + (-1+ i/(rw_env.num_states+1))
    policy[0] = policy[-1] = 0
    
    #create arrays to hold all the states and rewards
    states = np.zeros(n+1)
    rewards = np.zeros(n+1)
    
    values = np.zeros((ALPHAS.shape[0], rw_env.num_states))
    
    def greedy_policy(state):
        return np.random.choice(np.arange(0, len(action_list))) if np.random.random() < EPSILON else np.argmax(policy[state])
    
    episode_err = np.zeros((ALPHAS.shape[0], EPISODES))
    
    for e in range(EPISODES):
        T = float('inf')
        curr_time = 0
        action = greedy_policy(rw_env.curr_state)
        # print("Action is: ",action)
        while True:
            curr_time += 1
            if curr_time < T:
                
                stop,_,_,reward,state = rw_env.step(action_list[action])
            
                idx = curr_time % (n + 1)
                rewards[idx] = reward
                states[idx] = state
                
                if stop:
                    T = curr_time + 1
                else:
                    action = greedy_policy(rw_env.curr_state)
                    
            tau = curr_time - n
            
            if tau >= 0:
                G = 0.0
                for i in range((tau + 1), min(tau + n, T) + 1 ):
                    G += np.power(GAMMA, i - (tau + 1)) * rewards[i % (n + 1)]
                
                if (tau + n) < T:
                    state = int(states[(tau + n) % (n + 1)])
                    print("state: ",state)
                    G += np.power(GAMMA, n) * values[ :, state]
                    
                values[ :, int(states[tau % (n + 1)])] += ALPHAS * (G - values[ :, int(states[tau % (n + 1)])])

            if tau >= T - 1:
                break
            
        episode_err[ : , e] = np.sqrt(np.mean(np.square(values - policy), axis=-1))
        avg_err = np.mean(episode_err, axis=-1)
        print("average error is: ", avg_err)
        
    exit()
    return avg_err
          
def run_sim():
    
    NUM_STATES = 19
    EPISODES = 10
    RUNS = 100
    EPSILON = 0.1
    GAMMA = 1.0
    ALPHAS = np.linspace(0, 1, num=1001, endpoint=True)
    N_STEPS = np.power(2, np.arange(0, 10)) 
    
    rw = env(NUM_STATES)
    
    for n in N_STEPS:
        errors = np.zeros((ALPHAS.shape[0], RUNS))
        for r in range(RUNS):
            errors[ :, r] = n_stepTD(rw,ALPHAS,n,EPSILON,GAMMA,EPISODES)               
        errors = np.mean(errors, axis=-1)
        #Plot
        plt.plot(ALPHAS, errors, label="n = {}".format(n))
    

    
        

    
    plt.xlabel("Alphas")
    plt.ylabel("Average RMS error over {} states \nand first {} episodes".format(NUM_STATES, EPISODES))
    plt.title("Performance of n-step TD on 19-state Random Walk.")
    plt.ylim([0.25, 0.60])
    plt.legend(loc="upper left", bbox_to_anchor=(1.04, 1), ncol=1)
    plt.tight_layout()
    plt.savefig("Random_Walk_n_step_TD.png")
    plt.close()       
        
        
if __name__ == "__main__":
    run_sim()