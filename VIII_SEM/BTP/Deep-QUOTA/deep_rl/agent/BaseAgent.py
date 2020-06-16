#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import numpy as np

class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.evaluation_env = self.config.evaluation_env
        self.info = {}
        if self.evaluation_env is not None:
            self.evaluation_state = self.evaluation_env.reset()
            self.info['initial_state'] = True
            self.evaluation_return = 0

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)

    def load(self, filename):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)

    def evaluation_action(self, state):
        return None

    def deterministic_episode(self, max_steps=1e5):
        env = self.config.evaluation_env
        state = env.reset()
        total_rewards=0
        length = 0
        self.info['initial_state'] = True
        while True:
            action = self.evaluation_action(state)

            if action == None:
                print('Define evaluation action for your agent')
                exit(0)

            self.info['initial_state'] = False
            state, reward, done, _ = env.step(action)
            length += 1
            total_rewards += reward
            if done or length > max_steps:
                break
        return total_rewards

    def evaluation_episodes(self):
        rewards = []
        for ep in range(self.config.evaluation_episodes):
            rewards.append(self.deterministic_episode())
        return np.mean(rewards), np.std(rewards) / np.sqrt(len(rewards))