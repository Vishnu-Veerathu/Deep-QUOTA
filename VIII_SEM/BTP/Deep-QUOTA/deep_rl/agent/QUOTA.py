#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from ..utils import *
from .BaseAgent import *
import matplotlib.pyplot as plt

class QUOTA(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        if self.task.__class__.__name__ == 'grid':
            gridplt = 1 - self.task.blocks
            gridplt[19][0] = 2

            gridplt[0][0] = 3
            gridplt[0][19] = 4
            gridplt[19][19] = 3
            
            plt.imshow(gridplt)
            plt.show()

        self.network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.target_network = config.network_fn(self.task.state_dim, self.task.action_dim)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.target_network.load_state_dict(self.network.state_dict())
        self.policy = config.policy_fn()

        self.total_steps = 0
        self.state = self.task.reset()
        self.episode_reward = 0
        self.last_episode_rewards = 0

        self.quantile_weight = 1.0 / self.config.num_quantiles
        self.cumulative_density = self.network.tensor(
            (2 * np.arange(self.config.num_quantiles) + 1) / (2.0 * self.config.num_quantiles))

        self.option = np.random.randint(config.num_options)

        self.candidate_quantiles = self.network.tensor(config.candidate_quantiles).long()

        self.is_initial_state = True
        self.buffer = {'s': [], 'a': [], 'r': [], 'ns': [], 'd':[], 'o':[]}
        self.state_window = [self.state]
        self.action_window = []
        self.reward_window = []
        self.option_window = []
        self.last_end_state = None
        self.last_episode_length = None
        self.episode_length = 0

    def option_to_q_values(self, options, quantiles):
        config = self.config
        if config.smoothed_quantiles:
            if config.num_quantiles % config.num_options:
                raise Exception('Smoothed quantile options is not supported')
            quantiles = quantiles.view(quantiles.size(0), quantiles.size(1), config.num_options, -1)
            quantiles = quantiles.mean(-1)
            q_values = quantiles[:,:, options]
        else:
            print("ERROR with smoothed quantiles daww")
        return q_values

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1 - cond)

    def evaluation_action(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(np.stack([state]))
        quantile_values, option_values = self.network.predict(self.config.state_normalizer(state))
        greedy_options = torch.argmax(option_values, dim=-1)
        if self.config.option_type == 'constant_beta':
            dice = np.random.rand()
            start_new_option = self.info['initial_state'] or dice < self.config.target_beta
            if start_new_option:
                self.info['prev_option'] = greedy_options
            q_values = self.option_to_q_values(self.info['prev_option'], quantile_values)
        elif self.config.option_type is None:
            q_values = quantile_values.sum(-1)
        else:
            raise Exception('Unknown option type')

        q_values = q_values.cpu().detach().numpy()
        self.config.state_normalizer.unset_read_only()
        return np.argmax(q_values.flatten())

    def add_experience(self, exp):
        # print(exp)
        if len(self.buffer['s']) == self.config.buffer_size:
            for key in self.buffer.keys():
                self.buffer[key].pop(0)

        for key in self.buffer.keys():
            self.buffer[key].append(exp[key])
    
        
    def process_rewards(self, rewards):
        # print('Reward list I got->',rewards)
        proc_rwds = np.zeros(self.config.batch_size)
        for i in range(self.config.rollout_length):
            proc_rwds = self.config.discount * proc_rwds + rewards[:,-1 - i]
        # print('Processed reward list I shall return->',proc_rwds)
        return proc_rwds

    def train_network(self):
        config = self.config
        ids = np.random.choice(np.arange(len(self.buffer['s'])), size=config.batch_size)
        batch_states = np.asarray([self.buffer["s"][i] for i in ids])
        batch_actions = np.asarray([self.buffer["a"][i] for i in ids])
        batch_rewards = np.asarray([self.buffer["r"][i] for i in ids])
        batch_states_next = np.asarray([self.buffer["ns"][i] for i in ids])
        batch_dones = np.asarray([self.buffer["d"][i] for i in ids])
        batch_options = np.asarray([self.buffer["o"][i] for i in ids])
        
        quantile_values, option_values = self.network.predict(config.state_normalizer(batch_states))
        # print('Current state, quantile val and option val',batch_states,quantile_values,option_values)
        quantile_values_next, option_values_next = self.target_network.predict(config.state_normalizer(batch_states_next))
        
        batch_a_next = torch.argmax(quantile_values_next.sum(-1), dim=1)
        batch_o_next = torch.argmax(option_values_next, dim=1)
        
        # print('Next action, options',batch_a_next, batch_o_next)

        quantile_values_next, option_values_next = self.target_network.predict(config.state_normalizer(batch_states_next))
        returns = quantile_values_next[self.network.range(config.batch_size), batch_a_next,:].detach()
        # print('Quant next, option value next and retuns',quantile_values_next,option_values_next,returns)

        option_values_next = option_values_next.detach()
        
        option_returns = config.target_beta * option_values_next[self.network.range(config.batch_size),batch_o_next] + \
                        (1 - config.target_beta) * option_values_next[self.network.range(config.batch_size), batch_options]
        option_returns = option_returns.unsqueeze(1)
        # print('Option return values', option_returns)

        actions = self.network.tensor(batch_actions).long()
        quantile_values = quantile_values[self.network.tensor(np.arange(config.batch_size)).long(), actions,:]
        # print('quantile values for actions', quantile_values)
        terminals = self.network.tensor(batch_dones).unsqueeze(1)
        # print('terminals',terminals)
        # print('Batch_rewards',batch_rewards)
        rewards = self.process_rewards(batch_rewards)
        # print('Processed rewards', rewards)
        rewards = self.network.tensor(rewards).unsqueeze(1)
        returns = rewards + (config.discount ** config.rollout_length) * terminals * returns
        # print('processed returns', returns)
        option_returns = rewards + (config.discount ** config.rollout_length) * terminals * option_returns
        # print('target processed option returns', option_returns)
        option_values = option_values[self.network.range(config.batch_size), batch_options].unsqueeze(1)
        # print('prev option values', option_values)
        
        target_quantile_values = returns.t().unsqueeze(-1)
        diff = target_quantile_values - quantile_values
        # print('diff',diff)
        loss = self.huber(diff) * (self.cumulative_density.view(1, -1) - (diff.detach() < 0).float()).abs()
        # print('loss',loss)
        loss = loss.sum(0).mean(-1).mean(0)
    
        # print(option_values,option_returns)
        
        option_loss = (option_values - option_returns).pow(2).mul(0.5).mean()
        # print('batch_option loss', batch_option_loss)
        loss = loss + option_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.optimizer.step()
        
        
    def iteration(self):
        config = self.config
        one_step = True
        
        while len(self.buffer['s']) < self.config.batch_size or one_step:

            one_step = False

            # self.evaluation_episodes()
            quantile_value, option_value = self.network.predict(self.config.state_normalizer(self.state))
            # print('Op val', option_value)
            
            if np.random.rand() < config.behavior_beta or self.is_initial_state:
                
                if np.random.rand() < config.random_option_prob(1):
                    # print('Random option')
                    self.option = np.random.randint(config.num_options)
                else:
                    # print('Best option')
                    self.option = torch.argmax(option_value, dim=-1)
                    
            q_value = self.option_to_q_values(self.option, quantile_value)

            q_value = q_value.cpu().detach().numpy()

            action = self.policy.sample(q_value[0])

            self.action_window.append(action)
            self.option_window.append(int(self.option))

            next_state, reward, terminal, _ = self.task.step(action)
            self.episode_length += 1

            # print('Transition = ', self.state, self.task.state, action, next_state, terminal)
            # if action == 1 and (self.task.state != next_state or next_state[0] != -1):
            #     print('Gone1')
            #     exit(0)
            # elif action == 0 and (self.state[0] != next_state[0] - 1):
            #     print('Gone 2')
            #     exit(0)
            # elif self.state[0] == 10:
            #     print('Gone3')
            #     exit(0)
            # elif next_state != self.task.state:
            #     print('seesly wtaf')
            #     exit(0)
            self.state_window.append(next_state)
            self.reward_window.append(reward)

            if len(self.state_window) == config.rollout_length + 1:
                self.add_experience({'s':self.state_window[-self.config.rollout_length-1], 'a':self.action_window[-self.config.rollout_length],'r':self.reward_window.copy(),'ns':next_state,'d':1-terminal,'o':self.option_window[-config.rollout_length]})
                # print('Experience Addition->',self.state_window[-self.config.rollout_length-1],self.action_window[-self.config.rollout_length],self.reward_window,next_state,1-terminal,self.option_window[-config.rollout_length])
                # print('<-')
                self.state_window.pop(0)
                self.action_window.pop(0)
                self.reward_window.pop(0)
                self.option_window.pop(0)

            self.episode_reward += reward
            reward = config.reward_normalizer(reward)
            self.is_initial_state = terminal
            if terminal or self.episode_length > config.max_epsiode_length:
                self.last_episode_rewards = self.episode_reward
                self.last_end_state = next_state
                self.last_episode_length = self.episode_length
                self.episode_reward = 0
                self.episode_length = 0
                # print('Terminated in ', self.state)
                self.state = self.task.reset()
                self.state_window = [self.state]
                self.action_window = []
                self.reward_window = []
                self.option_window = []
                
            else:
                self.state = next_state

            # print(self.last_episode_rewards)
            
            self.policy.update_epsilon(1)
            self.total_steps += 1
            if self.total_steps % config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())
            
            if len(self.buffer['s']) >= self.config.batch_size and self.total_steps % config.train_frequency == 0:
                self.train_network()
            