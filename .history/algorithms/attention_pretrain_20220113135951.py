import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.loss import L1Loss
from torch.optim import Adam
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.policy.SAC_policy import SAC
from utils.policy.DDPG_policy import DDPG
from utils.critics import AttentionCritic
from utils.arguments import parse_args
from utils.attention_net import AttentionNet

MSELoss = torch.nn.MSELoss()
loss_l1 = torch.nn.L1Loss()

class AttentionPretrain(object):
    """
    Wrapper class for SAC agents with central attention critic in multi-agent
    task
    """

    def __init__(self, config, **kwargs):
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
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alphaT
        self.pi_lr = config.pi_lr
        self.q_lr = config.q_lr

        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.attention_dev = 'cpu'
        self.niter = 0
        self.device = torch.device("cuda")
        self.automatic_entropy_tuning = True
      
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.log_alpha], lr=self.q_lr)
        self.n_agents = config.nb_UAVs

        self.actor_local = DDPG(config)
        self.actor_target = DDPG(config)
        self.critic = AttentionCritic(config)
        self.target_critic = AttentionCritic(config)
        self.attention_net = AttentionNet(config)
        hard_update(self.target_critic, self.critic)
        self.critic1_optimizer = Adam(self.critic.critic1.parameters(), lr=config.q_lr,
                                      weight_decay=1e-3)
        # self.ExpLR_c1 = torch.optim.lr_scheduler.ExponentialLR(self.critic1_optimizer, gamma=0.99)
        
        self.critic2_optimizer = Adam(self.critic.critic2.parameters(), lr=config.q_lr,
                                      weight_decay=1e-3)
        # self.ExpLR_c2 = torch.optim.lr_scheduler.ExponentialLR(self.critic2_optimizer, gamma=0.99)
        self.policy_optimizer = Adam(self.actor_local.parameters(), lr=config.pi_lr,
                                     weight_decay=1e-3)
        # self.ExpLR_p = torch.optim.lr_scheduler.ExponentialLR(self.policy_optimizer, gamma=0.99)
        self.attention_optimizer = Adam([
            {'params': self.attention_net.parameters()},
            {'params': self.actor_local.parameters()},
        ], lr=config.pi_lr, weight_decay=1e-3)
        # self.ExpLR_attention = torch.optim.lr_scheduler.ExponentialLR(self.attention_optimizer, gamma=0.99)
        self.total_loss = 0

    @property
    def policy(self):
        return self.actor_local

    @property
    def target_policy(self):
        return self.actor_target

    def step(self, observations):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """
        attention_state ,postion_encode= self.attention_net.forward(observations)
        action, log_pi, = self.policy.forward(attention_state,postion_encode)
        return action, log_pi
        # return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
        #                                                        observations)]

    def take_optimisation_step(self, optimizer, network, loss, clipping_norm=None, retain_graph=False):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        if not isinstance(network, list): network = [network]
        optimizer.zero_grad()  # reset gradients to 0
        loss.backward(retain_graph=retain_graph)  # this calculates the gradients
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(net.parameters(),
                                               clipping_norm)  # clip gradients to help stabilise training
        optimizer.step()  # this applies the gradients                observations)]

    def update_all_targets(self, update_critic=True, update_actor=True):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        if update_critic:
            soft_update(self.target_critic, self.critic, self.tau)
        if update_actor:
            soft_update(self.actor_target, self.actor_local, self.tau)

    def prep_training(self, device='gpu'):
        self.critic.train()
        self.target_critic.train()
        self.actor_local.train()
        self.actor_target.train()
        self.attention_net.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            self.actor_local = fn(self.actor_local)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            self.actor_target = fn(self.actor_target)
            self.pol_dev = device
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device
        if not self.attention_dev == device:
            self.attention_net = fn(self.attention_net)
            self.attention_dev = device

        # todo 这个函数是干啥的？

    def prep_rollouts(self, device='cpu'):
        self.actor_local.eval()
        self.attention_net.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            self.actor_local = fn(self.actor_local)
            self.pol_dev = device
        if not self.attention_dev == device:
            self.attention_net = fn(self.attention_net)
            self.attention_dev = device

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
        save_dict = {
                     'actor_params': {'actor_local': self.actor_local.state_dict(),
                                      'actor_target': self.actor_target.state_dict(),
                                      'policy_optimizer': self.policy_optimizer.state_dict(),
                                      },
                     'attention_params': {'attention_net': self.attention_net.state_dict(),
                                          'attention_optimizer': self.attention_optimizer.state_dict(),
                                        },
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic1_optimizer': self.critic1_optimizer.state_dict(),
                                       'critic2_optimizer': self.critic2_optimizer.state_dict()}}
        torch.save(save_dict, filename)

    def test_parameters(self):
        print(self.agents.get_params()['policy'])

    @classmethod
    def init_from_save(cls, config, filename, load_critic=True, ):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(config)

        actor_params=save_dict['actor_params']

        instance.actor_local.load_state_dict(actor_params['actor_local'])
        instance.actor_target.load_state_dict(actor_params['actor_target'])
        instance.policy_optimizer.load_state_dict(actor_params['policy_optimizer'])

        attention_params = save_dict['attention_params']
        instance.attention_net.load_state_dict(attention_params['attention_net'])
        instance.attention_optimizer.load_state_dict(attention_params['attention_optimizer'])
        # print("It is time to init")
        # print(save_dict['agent_params'][0]['policy'])

        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic1_optimizer.load_state_dict(critic_params['critic1_optimizer'])
            instance.critic2_optimizer.load_state_dict(critic_params['critic2_optimizer'])
        return instance

    def pretrain_actor(self, sample):
        # print("update_policy!")
        share_state_batch, state_batch, action_batch, log_pi_batch, reward_batch, next_share_state_batch, next_state_batch, dones = sample
        share_state_batch = torch.tensor([item.cpu().detach().numpy() for item in share_state_batch]).cuda()
        # action_batch = torch.tensor([item.cpu().detach().numpy() for item in action_batch]).cuda()
        state_batch = torch.tensor([item.cpu().detach().numpy() for item in state_batch]).cuda()
        action_batch = torch.tensor([item.cpu().detach().numpy() for item in action_batch]).cuda()
       
        pi, _,  = self.step(share_state_batch,state_batch)
        
        # print(action_batch.shape)
        total_loss = 0
        for a_i, p_act, action in zip(range(self.n_agents), pi, action_batch):
            policy_loss = MSELoss(p_act, action)
            total_loss += policy_loss
        self.total_loss = total_loss
        # print(total_loss)
        self.take_optimisation_step(self.attention_optimizer, [self.actor_local,self.attention_net], total_loss, 5)
        return
    def pretrain_critic(self, sample):
        """
        Update central critic for all agents
        """
        # cpu设备上的相关操作
        # state_batch = [t.numpy() for t in state_batch]
        # state_batch = torch.Tensor(state_batch)
        print("update_critic!")
        share_state_batch,state_batch, action_batch, log_pi_batch, reward_batch, \
        next_share_state_batch, next_state_batch, dones = sample
        share_state_batch = torch.tensor([item.cpu().detach().numpy() for item \
            in share_state_batch]).cuda()
        
        state_batch = torch.tensor([item.cpu().detach().numpy() for item in state_batch]).cuda()
        
        next_share_state_batch = torch.tensor([item.cpu().detach().numpy() for item \
            in next_share_state_batch]).cuda()
        next_state_batch = torch.tensor([item.cpu().detach().numpy() for item \
            in next_state_batch]).cuda()
        
        action_batch = torch.tensor([item.cpu().detach().numpy() for item in action_batch]).cuda()
        # 下述操作为模仿的
        with torch.no_grad():
            next_attention_state ,next_pos= self.attention_net.forward(next_share_state_batch,next_state_batch)
            next_action, next_log_pi, = self.policy.forward(next_attention_state,next_pos)

            attention_state,pos = self.attention_net.forward(share_state_batch,state_batch)
            action, log_pi = self.policy.forward(attention_state,pos)
            qf1_next, qf2_next = self.target_critic(next_attention_state, next_pos,next_action)

            reward_batch = torch.tensor([item.cpu().detach().numpy() for item in reward_batch]).cuda()

            # print(reward_batch)
            min_qf_next_target = torch.min(qf1_next, qf2_next)
        qf1, qf2 = self.critic(attention_state, pos,action_batch)

        # print(next_log_pi)
        qf1_loss = 0
        qf2_loss = 0
        qf_loss = 0
        # print('reward')
        # print(reward_batch)
        # print(qf1)
        for a_i, q1, q2, nq in zip(range(self.n_agents), qf1, qf2, min_qf_next_target):
            target_q = (reward_batch[a_i].view(-1, 1) +
                        self.gamma * nq *
                        (1 - dones[a_i].view(-1, 1)))
            print("q1: ",end='')
            print(q1.mean())

            qf1_loss += loss_l1(q1, target_q.detach())
            qf2_loss += loss_l1(q2, target_q.detach())

        self.take_optimisation_step(self.critic1_optimizer, self.critic.critic1, qf1_loss, 5, retain_graph=True)
        self.take_optimisation_step(self.critic2_optimizer, self.critic.critic2, qf2_loss, 5, )
        return qf1_loss+qf2_loss
    def mc_pretrain_critic(self, sample):
        """
        Update central critic for all agents
        """
        # cpu设备上的相关操作
        # state_batch = [t.numpy() for t in state_batch]
        # state_batch = torch.Tensor(state_batch)
        print("update_critic!")
        share_state_batch,state_batch, action_batch, log_pi_batch, reward_batch,return_batch, \
        next_share_state_batch, next_state_batch, dones = sample
        share_state_batch = torch.tensor([item.cpu().detach().numpy() for item \
            in share_state_batch]).cuda()
        
        state_batch = torch.tensor([item.cpu().detach().numpy() for item in state_batch]).cuda()
        
        next_share_state_batch = torch.tensor([item.cpu().detach().numpy() for item \
            in next_share_state_batch]).cuda()
        next_state_batch = torch.tensor([item.cpu().detach().numpy() for item \
            in next_state_batch]).cuda()
        
        action_batch = torch.tensor([item.cpu().detach().numpy() for item in action_batch]).cuda()
        return_batch = torch.tensor([item.cpu().detach().numpy() for item in return_batch]).cuda()
        # 下述操作为模仿的
        with torch.no_grad():
            
            attention_state,pos = self.attention_net.forward(share_state_batch,state_batch)
            action, log_pi = self.policy.forward(attention_state,pos)
              
        qf1, qf2 = self.critic(attention_state, pos,action)

        # print(next_log_pi)
        qf1_loss = 0
        qf2_loss = 0
        qf_loss = 0
        # print('reward')
        # print(reward_batch)
        # print(qf1)
        for a_i, q1, q2 in zip(range(self.n_agents), qf1, qf2):
            target_q = (return_batch[a_i].view(-1, 1))
            if a_i==1:
                print("q1: ",end='')
                print(q1.mean())

            qf1_loss += loss_l1(q1, target_q.detach())
            qf2_loss += loss_l1(q2, target_q.detach())

        self.take_optimisation_step(self.critic1_optimizer, self.critic.critic1, qf1_loss, 5, retain_graph=True)
        self.take_optimisation_step(self.critic2_optimizer, self.critic.critic2, qf2_loss, 5, )
        return qf1_loss+qf2_loss

    def update_policy(self, sample, **kwargs):
        # print("update_policy!")
        share_state_batch,state_batch, action_batch, log_pi_batch, reward_batch,return_batch, \
        next_share_state_batch, next_state_batch, dones = sample
        share_state_batch = torch.tensor([item.cpu().detach().numpy() for item \
            in share_state_batch]).cuda()
        
        state_batch = torch.tensor([item.cpu().detach().numpy() for item in state_batch]).cuda()
        
        next_share_state_batch = torch.tensor([item.cpu().detach().numpy() for item \
            in next_share_state_batch]).cuda()
        next_state_batch = torch.tensor([item.cpu().detach().numpy() for item \
            in next_state_batch]).cuda()
        
        action_batch = torch.tensor([item.cpu().detach().numpy() for item in action_batch]).cuda()
        return_batch = torch.tensor([item.cpu().detach().numpy() for item in return_batch]).cuda()
        attention_state,pos = self.attention_net.forward(share_state_batch,state_batch)
        pi, log_pi = self.policy.forward(attention_state,pos)
        
        qf1_pi, qf2_pi = self.critic(attention_state,pos, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        print("min_qf_pi: ")
        print(min_qf_pi.mean())
        
        total_loss = 0
        for a_i, q in zip(range(self.n_agents),  min_qf_pi):
            policy_loss =(-q).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            total_loss += policy_loss
        
        # disable_gradients(self.critic)
        # self.target_entropy.cuda()
        # self.log_alpha
        # alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        print(total_loss.item())
        self.take_optimisation_step(self.policy_optimizer, self.actor_local, total_loss, 0.5)