from deep_rl import *

def configure_xgrid():
    """
    Configuring environment
    """
    rewards = np.zeros([20, 20])
    rewards[0][0] = 4
    rewards[0][19] = 80
    rewards[19][19] = 4

    blocks = np.zeros([20, 20])
    
    for i in range(20):
        blocks[i][10] = 1
        blocks[9][i] = 1

    blocks[4][10] = blocks[9][5] = blocks[9][15] = blocks[15][10] = 0

    starts = np.zeros([20, 20])
    starts[19][0] = 1

    done = np.zeros([20, 20])
    done[0][0] = 1
    done[0][19] = 1
    done[19][19] = 1

    optimals = np.array([[0,19]])

    return grid(rewards= rewards,blocks= blocks,starts= starts,done= done,optimals=optimals,epsilon= 0,illegal_reward=0)

def qrdqn_play_chain_MDP(cs,run,nature,**kwargs):
    kwargs.setdefault('tag',  nature + '_chain')
    kwargs.setdefault('gpu', 0)
    kwargs.setdefault('log_dir', '/home/yogesh/VIII_SEM/BTP/deep-quota/logs/QRDQN/'+nature+'-chain-' + str(cs) + '-run=' + str(run))
    kwargs.setdefault('max_steps', 2.5e4)
    kwargs.setdefault('num_quantiles', 3)
    kwargs.setdefault('option_type', 'constant_beta')
    kwargs.setdefault('num_options', 3)
    kwargs.setdefault('candidate_quantiles', np.linspace(0, kwargs['num_quantiles'] - 1, kwargs['num_options']))
    kwargs.setdefault('random_option_prob', LinearSchedule(1.0, 0, 1e4))
    kwargs.setdefault('target_beta', 0.0)
    kwargs.setdefault('behavior_beta', 0.0)
    kwargs.setdefault('smoothed_quantiles', True)
    kwargs.setdefault('q_epsilon', LinearSchedule(1.0, 0.0, 1e4))
    kwargs.setdefault('chi_priority',0.5)
    kwargs.setdefault('omega_priority',1.)
    kwargs.setdefault('beta_priority', LinearSchedule(0.6, 1, 1e5))
    kwargs.setdefault('epsilon_priority', 1e-3)
    kwargs.setdefault('batch_size', 64)
    kwargs.setdefault('buffer_size', 5000)
    kwargs.setdefault('chainlength',cs)
    kwargs.setdefault('max_epsiode_length', 1e4)
    kwargs.setdefault('train_frequency', 5)
    kwargs.setdefault('train_trajectories',5000)

    config = Config()
    config.merge(kwargs)
    os.makedirs(config.log_dir,exist_ok=True)

    if nature == 'opt':
        config.task_fn = lambda: Chain(config.chainlength)
    elif nature == 'pess':
        config.task_fn = lambda: LowerChain(config.chainlength)
    elif nature == 'monte':
        config.task_fn = lambda: MChain(config.chainlength)
    else:
        print('Ye kaunsa chain hai?')
        exit(0)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda state_dim, action_dim: \
        QuantileNet(action_dim, config.num_quantiles,
                                   FCBody(state_dim,(64,64)), gpu=kwargs['gpu'])
    config.policy_fn = lambda: GreedyPolicy(config.q_epsilon)
    config.state_normalizer = GridNormalizer(config.chainlength)
    config.reward_normalizer = IdentityNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 100
    config.rollout_length = 1
    config.gradient_clip=5
    config.replay_fn = lambda: Replay(memory_size=5000, batch_size=64)
    config.exploration_steps = 100
    run_episodes(QRDQN(config))

def play_chain_MDP(cs,run,nature,agent,**kwargs):
    kwargs.setdefault('tag', nature + '_chain')
    kwargs.setdefault('gpu', 0)
    kwargs.setdefault('log_dir', '/home/yogesh/VIII_SEM/BTP/deep-quota/logs/'+agent +'/' + nature + '-chain-' + str(cs) + '-run=' + str(run))
    kwargs.setdefault('max_steps', 2.5e4)
    kwargs.setdefault('num_quantiles', 3)
    kwargs.setdefault('option_type', 'constant_beta')
    kwargs.setdefault('num_options', 3)
    kwargs.setdefault('candidate_quantiles', np.linspace(0, kwargs['num_quantiles'] - 1, kwargs['num_options']))
    kwargs.setdefault('random_option_prob', LinearSchedule(1.0, 0, 1e4))
    kwargs.setdefault('target_beta', 0.000)
    kwargs.setdefault('behavior_beta', 0.000)
    kwargs.setdefault('smoothed_quantiles', True)
    kwargs.setdefault('q_epsilon', LinearSchedule(1.0, 0.0, 1e4))
    kwargs.setdefault('chi_priority',0.7)
    kwargs.setdefault('omega_priority',1.)
    kwargs.setdefault('beta_priority', LinearSchedule(0.6, 1, 1e5))
    kwargs.setdefault('epsilon_priority', 1e-3)
    kwargs.setdefault('batch_size', 64)
    kwargs.setdefault('buffer_size', 5000)
    kwargs.setdefault('chainlength',cs)
    kwargs.setdefault('max_epsiode_length', 1e4)
    kwargs.setdefault('train_frequency', 5)
    kwargs.setdefault('train_trajectories',6000)

    config = Config()
    config.merge(kwargs)
    os.makedirs(config.log_dir,exist_ok=True)

    if nature == 'opt':
        config.task_fn = lambda: Chain(config.chainlength)
    elif nature == 'pess':
        config.task_fn = lambda: LowerChain(config.chainlength)
    elif nature == 'monte':
        config.task_fn = lambda: MChain(config.chainlength)
    else:
        print('Ye kaunsa chain hai?')
        exit(0)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda state_dim, action_dim: \
        QLearningOptionQuantileNet(action_dim, config.num_quantiles, config.num_options,
                                   FCBody(state_dim,(64,64)), gpu=kwargs['gpu'])
    config.policy_fn = lambda: GreedyPolicy(config.q_epsilon)
    config.state_normalizer = GridNormalizer(config.chainlength)
    config.reward_normalizer = IdentityNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 100
    config.rollout_length = 1
    config.gradient_clip = 5
    if agent == 'DeepQUOTA':
        run_iterations(DeepQUOTA(config))
    elif agent == 'QUOTA':
        run_iterations(QUOTA(config))
    else:
        print('Error')

def qrdqn_play_grid_MDP(run,**kwargs):
    kwargs.setdefault('tag', play_grid_MDP.__name__)
    kwargs.setdefault('gpu', 0)
    kwargs.setdefault('log_dir', '/home/yogesh/VIII_SEM/BTP/deep-quota/logs/QRDQN/' + 'grid-run=' + str(run))
    kwargs.setdefault('max_steps', 2e6)
    kwargs.setdefault('num_quantiles', 15)
    kwargs.setdefault('option_type', 'constant_beta')
    kwargs.setdefault('num_options', 5)
    kwargs.setdefault('candidate_quantiles', np.linspace(0, kwargs['num_quantiles'] - 1, kwargs['num_options']))
    kwargs.setdefault('random_option_prob', ExpSchedule(1.0, 0.01,2e6))
    kwargs.setdefault('target_beta', 0.0001)
    kwargs.setdefault('behavior_beta', 0.0001)
    kwargs.setdefault('smoothed_quantiles', True)
    kwargs.setdefault('q_epsilon', ExpSchedule(1.0, 0.01,2e6))
    kwargs.setdefault('chi_priority',0.6)
    kwargs.setdefault('omega_priority',1.)
    kwargs.setdefault('beta_priority', LinearSchedule(0.6, 1, 2e6))
    kwargs.setdefault('epsilon_priority', 1e-3)
    kwargs.setdefault('batch_size', 64)
    kwargs.setdefault('buffer_size', 5000)
    kwargs.setdefault('max_epsiode_length', 1e4)
    kwargs.setdefault('train_frequency',2)

    config = Config()
    config.merge(kwargs)

    os.makedirs(config.log_dir,exist_ok=True)
    config.task_fn = lambda: configure_xgrid()
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda state_dim, action_dim: \
        QuantileNet(action_dim, config.num_quantiles, 
                                   FCBody(state_dim,(64,64,32)), gpu=kwargs['gpu'])
    config.policy_fn = lambda: GreedyPolicy(config.q_epsilon)
    config.state_normalizer = GridNormalizer(20)
    config.reward_normalizer = IdentityNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 100
    config.rollout_length = 3
    config.gradient_clip = 5
    config.replay_fn = lambda: Replay(memory_size=5000, batch_size=64)
    config.exploration_steps=100
    run_episodes(QRDQN(config))

def play_grid_MDP(run,agent,**kwargs):
    kwargs.setdefault('tag', play_grid_MDP.__name__)
    kwargs.setdefault('gpu', 0)
    kwargs.setdefault('log_dir', '/home/yogesh/VIII_SEM/BTP/deep-quota/logs/'+agent +'/' + 'grid-run=' + str(run))
    kwargs.setdefault('max_steps', 2e6)
    kwargs.setdefault('num_quantiles', 15)
    kwargs.setdefault('option_type', 'constant_beta')
    kwargs.setdefault('num_options', 5)
    kwargs.setdefault('candidate_quantiles', np.linspace(0, kwargs['num_quantiles'] - 1, kwargs['num_options']))
    kwargs.setdefault('random_option_prob', ExpSchedule(1.0, 0.01,2e6))
    kwargs.setdefault('target_beta', 0.0001)
    kwargs.setdefault('behavior_beta', 0.0001)
    kwargs.setdefault('smoothed_quantiles', True)
    kwargs.setdefault('q_epsilon', ExpSchedule(1.0, 0.01,2e6))
    kwargs.setdefault('chi_priority',0.6)
    kwargs.setdefault('omega_priority',1.)
    kwargs.setdefault('beta_priority', LinearSchedule(0.6, 1, 2e6))
    kwargs.setdefault('epsilon_priority', 1e-3)
    kwargs.setdefault('batch_size', 64)
    kwargs.setdefault('buffer_size', 5000)
    kwargs.setdefault('max_epsiode_length', 1e4)
    kwargs.setdefault('train_frequency',2)

    config = Config()
    config.merge(kwargs)

    os.makedirs(config.log_dir,exist_ok=True)
    config.task_fn = lambda: configure_xgrid()
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda state_dim, action_dim: \
        QLearningOptionQuantileNet(action_dim, config.num_quantiles, config.num_options,
                                   FCBody(state_dim,(64,64,32)), gpu=kwargs['gpu'])
    config.policy_fn = lambda: GreedyPolicy(config.q_epsilon)
    config.state_normalizer = GridNormalizer(20)
    config.reward_normalizer = IdentityNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 100
    config.rollout_length = 3
    config.gradient_clip = 5

    if agent == 'DeepQUOTA':
        run_iterations(DeepQUOTA(config))
    elif agent == 'QUOTA':
        run_iterations(QUOTA(config))

if __name__ == '__main__':
    run = [1, 2, 3]
    CL = [30]
    for i in CL:
        for j in run:
            print('-' * 10)
            print('Chain length = ',i,' , Run = ',j)
            play_grid_MDP(i,j,'monte','QUOTA')
            play_chain_MDP(i,j,'monte','DeepQUOTA')
            # play_chain_MDP(i,j,'pess','QUOTA')

    # font = {'family' : 'normal',
    #     'size'   : 6}
    # import matplotlib
    # matplotlib.rc('font', **font)
    # env = configure_xgrid()agent_name = agent.__class__.__name__
    
    # gridplt = np.ones((20, 20, 3))*255
    
    # for i in range(20):
    #     gridplt[i][10] = np.array([0,0,0])
    #     gridplt[9][i] = np.array([0,0,0])

    # gridplt[4][10] = gridplt[9][5] = gridplt[9][15] = gridplt[15][10] = np.array([255, 255, 255])
    
    # gridplt[19][0] = np.array([0, 255, 0])
    # gridplt[0][0] = gridplt[19][19] = np.array([255, 0, 0])
    # gridplt[0][19] = np.array([0, 0, 255])
    # plt.imshow(gridplt)
    # minor_ticks = np.arange(-.5, 20, 1)
    # major_ticks = np.arange(0, 20, 1)

    # ax = plt.gca()
    # ax.set_xticks(minor_ticks, minor = True)
    # ax.set_xticks(major_ticks)
    # ax.set_yticks(minor_ticks, minor = True)
    # ax.set_yticks(major_ticks)
    # ax.grid(which = 'minor',color='black', linestyle='-', linewidth=0.7)
    
    # ax.set_xticklabels(np.arange(1, 21, 1))
    # ax.set_yticklabels(np.arange(20, 0, -1))
    # plt.savefig('Montezuma Env.pdf',dpi = 200)
    # plt.show()