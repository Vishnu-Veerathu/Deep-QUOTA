#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np

class grid:
    def __init__(
        self, rewards, blocks, starts, done, optimals, epsilon, illegal_reward
    ):
        self.name = 'RoomGridWorld'
        self.state_dim = 2
        self.action_dim = 4
        self.rewards = rewards
        self.blocks = blocks
        self.starts = np.ndarray.flatten(starts)
        self.done = done
        self.rows = 20
        self.columns = 20
        self.epsilon = epsilon
        self.illegal_reward = illegal_reward
        self.counter = np.zeros([self.rows, self.columns])
        self.optimals = optimals

    def reset(self):
        self.state = np.array([19,0])
        self.counter[self.state[0]][self.state[1]] += 1
        return self.state

    def step(self, action):
        if np.random.random() < self.epsilon:
            action = np.random.randint(4)
        new_state = self.state.copy()
        out_of_bounds = False
        if action == 0:
            new_state[0] -= 1
            if new_state[0] < 0:
                out_of_bounds = True
        elif action == 1:
            new_state[1] += 1
            if new_state[1] >= self.columns:
                out_of_bounds = True
        elif action == 2:
            new_state[0] += 1
            if new_state[0] >= self.rows:
                out_of_bounds = True
        elif action == 3:
            new_state[1] -= 1
            if new_state[1] < 0:
                out_of_bounds = True
        # print(self.state,action,new_state)
        if out_of_bounds or self.blocks[new_state[0]][new_state[1]]:
            return self.state, 0, 0, {}
        self.counter[new_state[0]][new_state[1]] += 1
        self.state = new_state
        if self.done[new_state[0]][new_state[1]]:
            return self.state, self.rewards[new_state[0]][new_state[1]], self.done[new_state[0]][new_state[1]], None
    
        return new_state,0,self.done[new_state[0]][new_state[1]],None
        

    def counter_reset(self):
        self.counter = 0*self.counter

    def isopt(self):
        for i in self.optimals:
            if np.array_equal(i, self.state):
                return True
        return False

class Chain:
    def __init__(self, num_states, up_std=0.1, left_std=1.0):
        self.name = 'Opt-ChainWorld' + str(num_states)
        self.num_states = num_states
        self.state = np.array([0])
        self.action_dim = 2
        self.state_dim = 1
        self.up_std = up_std
        self.left_std = left_std
        self.counter = np.zeros(num_states + 1)
        self.optimals = np.array([num_states])

    def reset(self):
        self.state = np.array([0])
        return self.state

    def step(self, action):
        if action == 0:
            self.state = self.state + 1
            reward = np.random.randn() * self.left_std
            done = (self.state[0] == self.num_states)
            if done:
                self.counter[self.state[0]] += 1
                reward += 10.0
            return self.state, reward, done, None
        elif action == 1:
            self.counter[self.state[0]] += 1
            self.state = np.array([-1])
            return np.array([-1]), np.random.randn() * self.up_std, True, None
            
    def counter_reset(self):
        self.counter = 0 * self.counter
        
    def isopt(self):
        return self.state[0] == self.num_states

class LowerChain:
    def __init__(self, num_states, up_std=0.2, left_std=1.0):
        self.name = 'Pess-ChainWorld' + str(num_states)
        self.num_states = num_states
        self.state = np.array([0])
        self.action_dim = 2
        self.state_dim = 1
        self.up_std = up_std
        self.left_std = left_std
        self.counter = np.zeros(num_states + 1)
        self.optimals = np.array([num_states])

    def reset(self):
        self.state = np.array([0])
        return self.state

    def step(self, action):
        if action == 0:
            reward = -0.1
            self.state = self.state + 1
            done = (self.state[0] == self.num_states)
            if done:
                self.counter[self.state[0]] += 1
                reward = 10.0
            return self.state, reward, done, None
        elif action == 1:
            self.counter[self.state[0]] += 1
            self.state = np.array([-1])
            return np.array([-1]), np.random.randn(), True, None
            
    def counter_reset(self):
        self.counter = 0 * self.counter
        
    def isopt(self):
        return self.state[0] == self.num_states

class MChain:
    def __init__(self, num_states):
        self.name = 'Monte-ChainWorld' + str(num_states)
        self.num_states = num_states
        self.state = np.array([0])
        self.action_dim = 2
        self.state_dim = 1
        self.counter = np.zeros(num_states + 1)
        self.optimals = np.array([num_states])

    def reset(self):
        self.state = np.array([0])
        return self.state

    def step(self, action):
        if action == 0:
            reward = 0
            self.state = self.state + 1
            done = (self.state[0] == self.num_states)
            if done:
                self.counter[self.state[0]] += 1
                reward = 10.0
            return self.state, reward, done, None
        elif action == 1:
            self.counter[self.state[0]] += 1
            self.state = np.array([-1])
            return np.array([-1]), 1, True, None
            
    def counter_reset(self):
        self.counter = 0 * self.counter
        
    def isopt(self):
        return self.state[0] == self.num_states