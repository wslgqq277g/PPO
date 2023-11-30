import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Beta, Normal
import os
from hand_teleop.env.rl_env.point_net import PointNet
# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor_Beta(nn.Module):
    def __init__(self, args):
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.alpha_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.beta_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)
        return dist

    def mean(self, s):
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)  # The mean of the beta distribution
        return mean


class Actor_Gaussian(nn.Module):
    def __init__(self, args):
        super(Actor_Gaussian, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = self.max_action * torch.tanh(self.mean_layer(s))  # [-1,1]->[-max_action,max_action]
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s



class ActorCritic(nn.Module):
    def __init__(self, args):
        super(ActorCritic, self).__init__()

        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]
        self.hidden_width = args.hidden_width

        self.share_net = nn.Sequential(
                        nn.Linear(self.state_dim, self.hidden_width),
                        self.activate_func,
                        nn.Linear(self.hidden_width, self.hidden_width),
                        self.activate_func,
                    )

        # actor
        self.actor_mean_layer = nn.Linear(self.hidden_width, self.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))  # We use 'nn.Parameter' to train log_std automatically

        # critic
        self.critic_layer = nn.Linear(self.hidden_width, 1)

        
    def forward(self, s):
        h = self.share_net(s)
        action_mean = torch.tanh(self.actor_mean_layer(h))
        critic_value = self.critic_layer(h)
        return action_mean, critic_value

    def get_dist(self, s):
        action_mean, _ = self.forward(s)

        log_std = self.log_std.expand_as(action_mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(action_mean, std)  # Get the Gaussian distribution
        return dist
        
    
    # def act(self, state):

    #     action_mean = self.actor(state)
    #     cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
    #     dist = MultivariateNormal(action_mean, cov_mat)


    #     action = dist.sample()
    #     action_logprob = dist.log_prob(action)
        
    #     return action.detach(), action_logprob.detach()
    
    # def evaluate(self, state, action):

    #     action_mean = self.actor(state)
        
    #     action_var = self.action_var.expand_as(action_mean)
    #     cov_mat = torch.diag_embed(action_var).to(device)
    #     dist = MultivariateNormal(action_mean, cov_mat)
        
    #     # For Single Action Environments.
    #     if self.action_dim == 1:
    #         action = action.reshape(-1, self.action_dim)

    #     action_logprobs = dist.log_prob(action)
    #     dist_entropy = dist.entropy()
    #     state_values = self.critic(state)
        
    #     return action_logprobs, state_values, dist_entropy



class PPO_continuous_pc():
    def __init__(self, args):
        self.policy_dist = args.policy_dist
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_ac = args.lr_ac  # Learning rate of actor and critic

        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.critic_coef = args.critic_coef  # Critic coefficient

        
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.device = args.device
        self.use_visual_obs=args.use_visual_obs
        self.path=args.path
        if self.use_visual_obs==True:
            if len(self.path)==0:
                self.point_net=PointNet().to(self.device)
            else:
                self.point_net=PointNet(path=self.path).to(self.device)

        # if self.policy_dist == "Beta":
        #     self.actor = Actor_Beta(args).to(self.device)
        # else:
        #     self.actor = Actor_Gaussian(args).to(self.device)
        self.AC = ActorCritic(args).to(self.device)


        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_AC = torch.optim.Adam(({'params':self.AC.parameters()},\
                    {'params':self.point_net.parameters()}),lr=self.lr_ac, eps=1e-5)
        else:
            self.optimizer_AC = torch.optim.Adam(({'params':self.AC.parameters()},\
                    {'params':self.point_net.parameters()}), lr=self.lr_ac)


    def evaluate(self, s):  # When evaluating the policy, we only use the mean
        if isinstance(s, dict):
            s_state=torch.tensor(s['state'], dtype=torch.float).to(self.device)
            s_oracle=torch.tensor(s['oracle_state'], dtype=torch.float).to(self.device)
            s_camera_pc = torch.tensor(s['relocate-point_cloud'], dtype=torch.float).to(self.device)
            s_pc_feature = self.point_net(s_camera_pc.unsqueeze(0)).squeeze(0)
            s=torch.concat((s_pc_feature,s_state,s_oracle),dim=0)

            s = torch.unsqueeze(s, 0).to(self.device)
        else:
            s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)

        # s = torch.unsqueeze(torch.tensor(s, dtype=torch.float).to(self.device), 0)
        a, _ = self.AC(s)

        return a.detach().cpu().numpy().flatten()

    def choose_action(self, s):

        if isinstance(s, dict):
            s_state=torch.tensor(s['state'], dtype=torch.float).to(self.device)
            s_oracle=torch.tensor(s['oracle_state'], dtype=torch.float).to(self.device)
            s_camera_pc = torch.tensor(s['relocate-point_cloud'], dtype=torch.float).to(self.device)
            s_pc_feature = self.point_net(s_camera_pc.unsqueeze(0)).squeeze(0)
            s=torch.concat((s_pc_feature,s_state,s_oracle),dim=0)

            s = torch.unsqueeze(s, 0).to(self.device)
        else:
            s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)


        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)

        with torch.no_grad():
            dist = self.AC.get_dist(s)
            a = dist.sample()  # Sample the action according to the probability distribution
            a = torch.clamp(a, -1, 1)  # [-max,max]
            a_logprob = dist.log_prob(a)  # The log probability density of the action
        
        return a.cpu().numpy().flatten(), a_logprob.cpu().numpy().flatten()

    def update(self, replay_buffer, total_steps):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data
        if isinstance(s, list):
            for index,state in enumerate(s):
                s_state=torch.tensor(state['state'], dtype=torch.float).to(self.device)
                s_oracle=torch.tensor(state['oracle_state'], dtype=torch.float).to(self.device)
                s_camera_pc = torch.tensor(state['relocate-point_cloud'], dtype=torch.float).to(self.device)
                s_pc_feature = self.point_net(s_camera_pc.unsqueeze(0)).squeeze(0)
                state_f=torch.concat((s_pc_feature,s_state,s_oracle),dim=0)
                state_f = torch.unsqueeze(state_f, 0).to(self.device)
                if index==0:
                    state_f_all=state_f
                else:
                    state_f_all=torch.concat([state_f_all,state_f],dim=0)

        if isinstance(s_, list):
            for index,state in enumerate(s_):
                s_state=torch.tensor(state['state'], dtype=torch.float).to(self.device)
                s_oracle=torch.tensor(state['oracle_state'], dtype=torch.float).to(self.device)
                s_camera_pc = torch.tensor(state['relocate-point_cloud'], dtype=torch.float).to(self.device)
                s_pc_feature = self.point_net(s_camera_pc.unsqueeze(0)).squeeze(0)
                state_f=torch.concat((s_pc_feature,s_state,s_oracle),dim=0)
                state_f = torch.unsqueeze(state_f, 0).to(self.device)
                if index==0:
                    state_f_all_=state_f
                else:
                    state_f_all_=torch.concat([state_f_all_,state_f],dim=0)

        state_f_all = state_f_all.to(self.device)
        a = a.to(self.device)
        a_logprob = a_logprob.to(self.device)
        r = r.to(self.device)
        state_f_all_ = state_f_all_.to(self.device)
        dw = dw.to(self.device)
        done = done.to(self.device)
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            _, vs = self.AC(state_f_all)
            _, vs_ = self.AC(state_f_all_)

            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs

            for delta, d in zip(reversed(deltas.cpu().flatten().numpy()), reversed(done.cpu().flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(self.device)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                # s, a, a_logprob, r, s_, dw, done

                # print(index)
                # print(type(index))
                for order,num in enumerate(index):
                    state=s[num]
                    s_state = torch.tensor(state['state'], dtype=torch.float).to(self.device)
                    s_oracle = torch.tensor(state['oracle_state'], dtype=torch.float).to(self.device)
                    s_camera_pc = torch.tensor(state['relocate-point_cloud'], dtype=torch.float).to(self.device)
                    s_pc_feature = self.point_net(s_camera_pc.unsqueeze(0)).squeeze(0)
                    state_f = torch.concat((s_pc_feature, s_state, s_oracle), dim=0)
                    state_f = torch.unsqueeze(state_f, 0).to(self.device)
                    if order==0:
                        state_f_all=state_f
                    else:
                        state_f_all=torch.concat([state_f_all,state_f],dim=0)


                # s_state = torch.tensor(state_['state'], dtype=torch.float).to(self.device)
                # s_oracle = torch.tensor(state_['oracle_state'], dtype=torch.float).to(self.device)
                # s_camera_pc = torch.tensor(state_['relocate-point_cloud'], dtype=torch.float).to(self.device)
                # s_pc_feature = self.point_net(s_camera_pc.unsqueeze(0)).squeeze(0)
                # state_f_ = torch.concat((s_pc_feature, s_state, s_oracle), dim=0)
                # # state_f = torch.unsqueeze(state_f, 0).to(self.device)




                # dist_now = self.AC.get_dist(state_f)
                dist_now = self.AC.get_dist(state_f_all)
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index])
                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]

                _, v_s = self.AC(state_f_all)
                critic_loss = F.mse_loss(v_target[index], v_s)
                
                loss = -torch.min(surr1, surr2) + self.critic_coef * critic_loss - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                
                # Update actor
                self.optimizer_AC.zero_grad()
                loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.AC.parameters(), 0.5)
                self.optimizer_AC.step()
        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_ac_now = self.lr_ac * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_AC.param_groups:
            p['lr'] = lr_ac_now


    def save(self, directory):
        isExists = os.path.exists(directory)
        if isExists == False:
            os.mkdir(directory)

        torch.save(self.AC.state_dict(), directory + '/AC.pth')

if __name__ == '__main__':
    pass
