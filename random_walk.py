from threading import currentThread
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib import polynomial

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

    def step(self, action):
        
        reward = 0
        stop = False

        self.curr_state += action
        
        if self.curr_state <= self.terminal_state_2:
            reward = 0
            stop = True
        elif self.curr_state >= self.terminal_state_1:
            reward = 1
            stop = True

        return stop, reward, self.curr_state

'''
Does the N-Step TD method for an alpha value using the random walk example from 6.2
Inputs: rw_env, ALPHAS, n (for n_steps), EPSILON, GAMMA, EPISODES

''' 
def N_Step_TD(rw_env, n, ALPHAS, EPSILON=0.1,  GAMMA=0.95, EPISODES=10,TD_SUM = False):
      
    #policy based on bellman equation
    true_val = np.zeros(rw_env.num_states)
    for i in range(1, rw_env.num_states+1):
        true_val[i-1] = i / (rw_env.num_states+1) + (-1 + i/(rw_env.num_states+1))
    
    #action list
    right = 1
    left = -1
    action_list = [right, left]
    
    #State and Reward memory arrays and Value array to hold all values for each state
    states = np.zeros(n+1)
    rewards = np.zeros(n+1)
    values = np.zeros((ALPHAS.shape[0], rw_env.num_states))

    policy = np.ones(20)
    
    #greedy policy
    def greedy_policy(state):
        return np.random.choice(np.arange(0, len(action_list))) if np.random.random() < EPSILON else np.argmax(policy[state])
    
    episode_err = np.zeros((ALPHAS.shape[0], EPISODES))
    for episode in range(EPISODES):
        #reset environment and choose 
        rw_env.reset()
        states[0] = rw_env.curr_state
        action = greedy_policy(rw_env.curr_state)
        TD_ERR_SUM = 0
        
        curr_time = 0
        T = float('inf')
        
        while True:
         
            if curr_time < T:
                stop, reward, curr_state = rw_env.step(action_list[action])

                idx = curr_time % (n + 1)
                rewards[idx] = reward
                states[idx] = curr_state

                if stop:
                    T = curr_time 
                else:
                    action = greedy_policy(rw_env.curr_state)
            
            tau = curr_time - n

            if tau >= 0:
                
                G = 0.0

                
                if TD_SUM == True:
                    for i in range((tau + 1), min(tau + n, T) + 1 ):
                        G += pow(GAMMA, i - (tau - 1)) * rewards[i % (n + 1)]
                    if (tau + n) < T:
                        G += pow(GAMMA, n) * values[ :, int(states[(tau + n) % (n + 1)])]
                    TD_ERR_SUM += (G - values[ :, int(states[tau % (n + 1)])])
                    values[ :, int(states[tau % (n + 1)])] += ALPHAS * (TD_ERR_SUM)

                else:
                    for i in range((tau + 1), min(tau + n, T) + 1 ):
                        G += pow(GAMMA, i - (tau - 1)) * rewards[i % (n + 1)]
                    if (tau + n) < T:
                        G += pow(GAMMA, n) * values[ :, int(states[(tau + n) % (n + 1)])]
                    values[ :, int(states[tau % (n + 1)])] += ALPHAS * (G - values[ :, int(states[tau % (n + 1)])])

    
            if tau >= T - 1:
                break
            curr_time += 1
            
        
    
        #Calculate RMS and average the errors over the states.
        episode_err[ : , episode] = np.sqrt(np.mean(np.square(values - true_val), axis=-1))

    #Average errors over episodes and return.
    return np.mean(episode_err, axis=-1)

def run_sim():
    
    plt.figure()
    
    NUM_STATES=19
    rw_env = env(num_states=NUM_STATES)
    
    GAMMA = 0.99
    EPISODES = 10
    EPSILON = 0.1
    RUNS = 100
    ALPHAS = np.linspace(0, 1, num=1001, endpoint=True)
    N_STEPS = np.power(2, np.arange(0, 10)) 
    
    for n in N_STEPS:
        print("N: ", n)
        errors = np.zeros((ALPHAS.shape[0], RUNS))
        errorsTDS = np.zeros((ALPHAS.shape[0], RUNS))
        for r in range(RUNS):
            # errors[ :, r] = N_Step_TD(rw_env, n, ALPHAS, EPSILON, GAMMA, EPISODES,TD_SUM=False)
            errorsTDS[ :, r] = N_Step_TD(rw_env, n, ALPHAS, EPSILON, GAMMA, EPISODES,TD_SUM=True)
        
        #Average the errors over the runs.
        # errors = np.mean(errors, axis=-1)
        #For TD SUM
        errorsTDS = np.mean(errorsTDS, axis=-1)
        
        #Plot
        # plt.plot(ALPHAS, errors, label="n = {}".format(n))
        plt.plot(ALPHAS, errorsTDS, label="n2 = {}".format(n))
      
    print("Done.")
    
    plt.title("N-step TD on {} - state Random Walk".format(NUM_STATES))
    plt.xlabel("Alphas")
    plt.ylabel("Average RMS error over {} states \nand first {} episodes".format(NUM_STATES, EPISODES))
    plt.ylim([0.25,3.5])
    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1), ncol=1)
    plt.tight_layout()
    plt.savefig("Random_Walk_n_step_TD_SUM.png")
    plt.close()

if __name__ == "__main__":
    run_sim()