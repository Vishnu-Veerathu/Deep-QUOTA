#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import pickle
import os
import datetime
import torch
import sys
try:
    # python >= 3.5
    from pathlib import Path
except:
    # python == 2.7
    from pathlib2 import Path

def random_seed():
    np.random.seed()
    torch.manual_seed(np.random.randint(int(1e6)))

def run_episodes(agent):
    random_seed()
    config = agent.config
    ep = 0
    rewards = []
    steps = []
    agent_name = agent.__class__.__name__
    print("Agent= ",agent_name)
    while True:
        ep += 1
        reward, step, last_state = agent.episode()
        rewards.append(reward)
        steps.append(step)
        print('Episode # ', ep, ', Episode reward=', reward, ' , Terminal state=', last_state, 'Epsiode length=', step)

        if ep % config.iteration_log_interval == 0:
            np.save(config.log_dir + '/train_rewards.npy',np.array(rewards))
            np.save(config.log_dir + '/opt-counter.npy', agent.task.counter)

        if ep > config.train_trajectories:
            break
        
    np.save(config.log_dir + '/train_rewards.npy',np.array(rewards))
    np.save(config.log_dir + '/opt-counter.npy', agent.task.counter)

    return steps,rewards

def run_iterations(agent):
    random_seed()
    config = agent.config
    iteration = 0
    steps = []
    rewards = []
    episode_num = 0
    agent_name = agent.__class__.__name__
    print("Agent= ", agent_name)
    while True:
        agent.iteration()
        if agent.is_initial_state:
            episode_num +=1
            steps.append(agent.total_steps)
            rewards.append(agent.last_episode_rewards)

            print('Episode # ', episode_num, ' , Step # ', iteration, ' , Episode reward=', agent.last_episode_rewards, ' , Terminal state=', agent.last_end_state, 'Epsiode length=', agent.last_episode_length)
            # print('Option values->',agent.network.predict(config.state_normalizer())[1].cpu().detach().numpy()[0])
        if iteration % (config.iteration_log_interval * 100) == 0:
            np.save(config.log_dir + '/train.npy',np.array(rewards))
            np.save(config.log_dir + '/opt-counter.npy', agent.task.counter)
            optval = np.zeros([config.chain_length,config.num_options])
            for i in range(config.chain_length):
                state = np.array([[i]])
                optval[i] = to_numpy(agent.network.predict(config.state_normalizer(state))[1][0])
                np.save(config.log_dir+'/opt-values.npy',optval)

            # agent.save(config.log_dir + '/%s-%s-%s_run='%(agent_name, config.tag, agent.task.name)+str(run)+'-model.bin')

        iteration += 1

        if episode_num > config.train_trajectories:
            print('Closing now')
            break
    
    np.save(config.log_dir + '/train.npy',np.array(rewards))
    np.save(config.log_dir + '/opt-counter.npy', agent.task.counter)
    optval = np.zeros([config.chain_length,config.num_options])
    for i in range(config.chain_length):
        state = np.array([[i]])
        optval[i] = to_numpy(agent.network.predict(config.state_normalizer(state))[1][0])
        np.save(config.log_dir+'/opt-values.npy',optval)
    
    return steps, rewards

def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")

def sync_grad(target_network, src_network):
    for param, src_param in zip(target_network.parameters(), src_network.parameters()):
        param._grad = src_param.grad.clone()

def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def set_one_thread():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

def get_default_log_dir(name):
    return './log/%s-%s' % (name, get_time_str())

def to_numpy(x):
    return x.detach().cpu().numpy()

def console(fn):
    def wrapper(*args, **kwargs):
        if 'id' in kwargs:
            kwargs['ids'] = [kwargs['id']]
            del kwargs['id']
        if 'ids' in kwargs:
            ids = kwargs['ids']
            if len(ids) + 1 != len(sys.argv):
                return
            valid = True
            for i, id in enumerate(ids):
                if id != int(sys.argv[i + 1]):
                    valid = False
                    break
            if not valid:
                return
        fn(*args, **kwargs)
    return wrapper
