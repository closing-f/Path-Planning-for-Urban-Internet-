    # def update_critic(self, sample, soft=True, logger=None, **kwargs):
    #     """
    #      sac Update central critic for all agents
    #     """
    #     # cpu设备上的相关操作
    #     # state_batch = [t.numpy() for t in state_batch]
    #     # state_batch = torch.Tensor(state_batch)
    #     # print("update_critic!")
    #     state_batch, action_batch, log_pi_batch, reward_batch, next_state_batch, dones = sample
    #     next_state_batch = torch.tensor([item.cpu().detach().numpy() for item in next_state_batch]).cuda()
    #     state_batch = torch.tensor([item.cpu().detach().numpy() for item in state_batch]).cuda()
    #     action_batch = torch.tensor([item.cpu().detach().numpy() for item in action_batch]).cuda()
    #     reward_batch = torch.tensor([item.cpu().detach().numpy() for item in reward_batch]).cuda()
    #     # 下述操作为模仿的
    #     with torch.no_grad():
    #         next_attention_state = self.attention_net.forward(next_state_batch)
    #         next_action, next_log_pi = self.policy.forward(next_attention_state)
            
    #         qf1_next, qf2_next = self.target_critic(next_attention_state,next_action)
    #         min_qf_next_target = torch.min(qf1_next, qf2_next)- self.alpha * next_log_pi
    #         attention_state = self.attention_net.forward(state_batch)
        
    #     # print(min_qf_next_target)
    #     qf1, qf2 = self.critic(next_attention_state, action_batch)
    #     qf1_loss = 0
    #     qf2_loss = 0
    #     for a_i, q1, q2, nq in zip(range(self.n_agents), qf1, qf2, min_qf_next_target):
    #         target_q = (reward_batch[a_i].view(-1, 1) +
    #                     self.gamma * nq *
    #                     (1 - dones[a_i].view(-1, 1)))
            
    #         qf1_loss += MSELoss(q1, target_q.detach())
    #         qf2_loss += MSELoss(q2, target_q.detach())
    #     self.take_optimisation_step(self.critic1_optimizer, self.critic.critic1, qf1_loss, 5, retain_graph=True)
    #     self.take_optimisation_step(self.critic2_optimizer, self.critic.critic2, qf2_loss, 5, )
        
    # def update_policy(self, sample, **kwargs):
    #     # print("update_policy!")
    #     share_state_batch,state_batch, action_batch, log_pi_batch, reward_batch, \
    #     next_share_state_batch, next_state_batch, dones = sample
    #     share_state_batch = torch.tensor([item.cpu().detach().numpy() for item \
    #         in share_state_batch]).cuda()
        
    #     state_batch = torch.tensor([item.cpu().detach().numpy() for item in state_batch]).cuda()
        
    #     next_share_state_batch = torch.tensor([item.cpu().detach().numpy() for item \
    #         in next_share_state_batch]).cuda()
    #     next_state_batch = torch.tensor([item.cpu().detach().numpy() for item \
    #         in next_state_batch]).cuda()
        
    #     action_batch = torch.tensor([item.cpu().detach().numpy() for item in action_batch]).cuda()
    #     attention_state,pos = self.attention_net.forward(share_state_batch,state_batch)
    #     pi, log_pi = self.policy.forward(attention_state,pos)
    #     # print(pi)
    #     qf1_pi, qf2_pi = self.critic(attention_state,pos,pi)
    #     min_qf_pi = torch.min(qf1_pi, qf2_pi)
    #     print("min_qf_pi: ")
    #     print(min_qf_pi.mean())
    #     # log_pi_sum = log_pi.sum(2, keepdim=True) / 2
    #     # print(log_pi_sum.shape)
    #     # print(min_qf_pi.shape)
    #     total_loss = 0
    #     for a_i, log_pi, q in zip(range(self.n_agents), log_pi, min_qf_pi):
    #         policy_loss =((self.alpha * log_pi)-q).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
    #         total_loss += policy_loss
        

    #     # self.target_entropy.cuda()
    #     # self.log_alpha
        
    #     self.take_optimisation_step(self.policy_optimizer, self.actor_local, total_loss, 5, retain_graph=True)
    #     alpha_loss=0
    #     for i ,log_pi in zip(range(self.n_agents),log_pi):
    #         alpha_loss += -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
    #     self.take_optimisation_step(self.alpha_optim, self.log_alpha, alpha_loss )
    #     self.alpha = self.log_alpha.exp()
