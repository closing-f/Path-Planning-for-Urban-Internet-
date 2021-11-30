import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.critics import AttentionCritic
from utils.arguments import parse_args

MSELoss = torch.nn.MSELoss()


class AttentionPPO(object):
    """
    Wrapper class for SAC agents with central attention critic in multi-agent
    task
    """
    def __init__(self,
                 **kwargs):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
            sa_size (list of (int, int)): Size of state and action space for
                                          each agent
            gamma (float): Discount factor
            tau (float): Target update rate
            pi_lr (float): Learning rate for policy
            q_lr (float): Learning rate for critic
            reward_scale (float): Scaling for reward (has effect of optimal
                                  policy entropy)
            hidden_dim (int): Number of hidden dimensions for networks
        """
        config = parse_args()
        self.nagents = config.nb_UAVs

        self.agents = AttentionAgent(policy_type='TRPO')
        self.critic = AttentionCritic()
        self.target_critic = AttentionCritic()
        #
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.q_lr,
                                     weight_decay=1e-3)
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alphaT
        self.pi_lr = config.pi_lr
        self.q_lr = config.q_lr
        self.reward_scale = config.reward_scale
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0
        self.device = torch.device("cpu")
        self.automatic_entropy_tuning = True

        self.target_entropy = -torch.prod(torch.Tensor(config.dim_action).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.log_alpha], lr=self.q_lr)

        self.total_loss = 0

    @property
    def policies(self):
        return self.agents.policy

    @property
    def target_policies(self):
        return self.agents.target_policy

    def step(self, share_obs,observations):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """
        return self.agents.step(share_obs,observations)
        # return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
        #                                                        observations)]

    def update_critic(self, sample, soft=True, logger=None, **kwargs):

        """
        Update central critic for all agents
        """

        print("update_critic!")
        state_batch, action_batch, _, reward_batch, next_state_batch, dones = sample

        # 从replay buffer中取出来的不是tensor，需要转换成tensor,下述为转换步骤
        next_state_batch = [t.numpy() for t in next_state_batch]
        next_state_batch = np.array(next_state_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)

        state_batch = [t.numpy() for t in state_batch]
        state_batch = np.array(state_batch)
        state_batch = torch.FloatTensor(state_batch)

        action_batch = [t.numpy() for t in action_batch]
        action_batch = np.array(action_batch)
        action_batch = torch.FloatTensor(action_batch)

        # 下述操作为模仿的
        with torch.no_grad():
            next_action, next_log_pi, next_other_all_values, next_s_encoding = self.policies.policy_sample(
                next_state_batch, update_critic=True)
            action, log_pi, other_all_values, s_encoding = self.policies.policy_sample(state_batch,
                                                                                       update_critic=True)

            qf1_next, qf2_next = self.target_critic(next_other_all_values, next_s_encoding,
                                                                  next_action)


            reward_batch = [t.numpy() for t in reward_batch]
            reward_batch = np.array(reward_batch)
            reward_batch = torch.FloatTensor(reward_batch)

            min_qf_next_target = torch.min(qf1_next, qf2_next) - self.alpha * next_log_pi

        qf1, qf2 = self.critic(other_all_values, s_encoding, action_batch)
        qf1_loss = 0
        qf2_loss = 0
        for a_i, q1, q2, nq in zip(range(self.nagents), qf1, qf2, min_qf_next_target):
            target_q = (reward_batch[a_i].view(-1, 1) +
                        self.gamma * nq *
                        (1 - dones[a_i].view(-1, 1)))

            # target_q不会参与反向传播，q1,q2应该会反向传播
            qf1_loss += MSELoss(q1, target_q.detach())
            qf2_loss += MSELoss(q2, target_q.detach())

        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        print(qf_loss)
        qf_loss.backward()
        self.critic_optimizer.step()

    def update_all_targets(self, update_critic=True, update_actor=True):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        if update_critic:
            soft_update(self.target_critic, self.critic, self.tau)
        if update_actor:
            soft_update(self.agents.target_policy, self.agents.policy, self.tau)

    def prep_training(self, device='gpu'):
        self.critic.train()
        self.target_critic.train()
        self.agents.policy.train()
        self.agents.target_policy.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            self.agents.policy = fn(self.agents.policy.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            self.agents.target_policy = fn(self.agents.target_policy)
            self.pol_dev = device
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device

    # todo 这个函数是干啥的？
    def prep_rollouts(self, device='cpu'):
        self.agents.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            self.agents.policy = fn(self.agents.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        # print("It is time to save")
        # print(self.agents.get_params()['policy'])
        # print(self.agents.get_params()['policy']['policy_net.0.weight'])
        # print(self.agents.get_params()['policy']['mean_linear.0.weight'])

        # print(self.agents.get_params()['policy']['policy_net.0.weight'])
        # print(self.agents.get_params()['policy']['shared_net_pi.2.weight'])
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [self.agents.get_params()],
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict()}}
        torch.save(save_dict, filename)
    def test_parameters(self):
        print(self.agents.get_params()['policy'])
    @classmethod
    def init_from_env(cls, env, config, **kwargs):
        """
        Instantiate instance of this class from multi-agent environment

        env: Multi-agent Gym environment
        gamma: discount factor
        tau: rate of update for target networks
        lr: learning rate for networks
        hidden_dim: number of hidden dimensions for networks
        """
        sa_size = []
        obs_dim_single_all = (config.max_nb_UAVs * config.uav_obs_dim) + 2 + (config.nb_PoIs * config.poi_dim)
        for i in range(config.nb_UAVs):
            sa_size.append((obs_dim_single_all, config.dim_action))
        # for acsp, obsp in zip(range(arglist.)):
        #     agent_init_params.append({'num_in_pol': obsp.shape[0],
        #                               'num_out_pol': acsp.n})
        #     sa_size.append((obsp.shape[0], acsp.n))

        init_dict = {'gamma': config.gamma, 'tau': config.tau,
                     'pi_lr': config.pi_lr, 'q_lr': config.q_lr,
                     'alphaT': config.alphaT,
                     'reward_scale': config.reward_scale,
                     'pol_hidden_dim': config.pol_hidden_dim,
                     'critic_hidden_dim': config.critic_hidden_dim,
                     'attend_heads': config.attend_heads,
                     'sa_size': sa_size}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance
    '''
    @classmethod
    def init_from_save(cls, filename, load_critic=True):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        instance.agents.load_params(save_dict['agent_params'][0])
        # print("It is time to init")
        # print(save_dict['agent_params'][0]['policy'])

        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
        return instance
    '''
    @classmethod
    def init_from_save(cls, config, filename, load_critic=True, ):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(config)

        actor_params=save_dict['actor_params']
        # print(actor_params['actor_local'])
        instance.actor_local.load_state_dict(actor_params['actor_local'])
        instance.actor_target.load_state_dict(actor_params['actor_target'])
        # instance.policy_optimizer.load_state_dict(actor_params['policy_optimizer'])

        attention_params = save_dict['attention_params']
        instance.attention_net.load_state_dict(attention_params['attention_net'])
        # instance.attention_optimizer.load_state_dict(attention_params['attention_optimizer'])
        # print("It is time to init")
        # print(save_dict['agent_params'][0]['policy'])

        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            # print(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            # instance.critic1_optimizer.load_state_dict(critic_params['critic1_optimizer'])
            # instance.critic2_optimizer.load_state_dict(critic_params['critic2_optimizer'])
        return instance




    def update_policy(self, sample, **kwargs):
        print("update_policy!")
        state_batch, action_batch, batch_log_probs, reward_batch, next_state_batch, dones = sample

        state_batch = [t.numpy() for t in state_batch]
        state_batch = torch.FloatTensor(state_batch)

        action_batch = [t.numpy() for t in action_batch]
        action_batch = torch.FloatTensor(action_batch)

        batch_log_probs = [t.numpy() for t in batch_log_probs]
        batch_log_probs = torch.FloatTensor(batch_log_probs)

        pi, log_pi, other_all_values, s_encoding_tensor = self.policies.policy_sample(state_batch, update_critic=True)

        qf1_pi, qf2_pi = self.critic(other_all_values, s_encoding_tensor, action_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        ratios = torch.exp(log_pi - batch_log_probs)
        surr1 = ratios * min_qf_pi
        surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * min_qf_pi
        total_loss = 0

        disable_gradients(self.critic)
        actor_loss = (-torch.min(surr1, surr2)).mean()
        self.agents.policy_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        print(actor_loss)
        self.agents.policy_optimizer.step()
        enable_gradients(self.critic)