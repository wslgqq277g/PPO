import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Beta, Normal
import os
from model.VAE import AutoencoderPN, depth_to_point_cloud
import numpy as np
from model.point_net import EncoderDecoder


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

def chamfer_loss(point_set1, point_set2):
    # point_set1: (B, N, D)
    # point_set2: (B, M, D)

    # 计算点集1中每个点到点集2的最小距离
    point_set1_expand = point_set1.unsqueeze(2)  # (B, N, 1, D)
    point_set2_expand = point_set2.unsqueeze(1)  # (B, 1, M, D)
    dist1 = torch.sum((point_set1_expand - point_set2_expand) ** 2, dim=-1)  # (B, N, M)
    min_dist1, _ = torch.min(dist1, dim=2)  # (B, N)

    # 计算点集2中每个点到点集1的最小距离
    dist2 = torch.sum((point_set1_expand - point_set2_expand) ** 2, dim=-1)  # (B, N, M)
    min_dist2, _ = torch.min(dist2, dim=1)  # (B, M)

    # 计算Chamfer距离
    chamfer_dist = torch.mean(min_dist1, dim=1) + torch.mean(min_dist2, dim=1)  # (B,)
    chamfer_loss = torch.mean(chamfer_dist)

    return chamfer_loss
def ChamLoss(coarse, fine, gt, alpha):
    return chamfer_loss(coarse, gt) + alpha * chamfer_loss(fine, gt)


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
        self.log_std = nn.Parameter(
            torch.zeros(1, args.action_dim))  # We use 'nn.Parameter' to train log_std automatically

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

class PPO_continuous_pc(nn.Module):
    def __init__(self, args):
        super(PPO_continuous_pc, self).__init__()
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
        self.cham_coef = args.cham_coef
        self.state_coef = args.state_coef
        self.camera_intrinsic = args.camera_matrix

        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.device = args.device
        self.use_visual_obs = args.use_visual_obs
        self.path = args.path
        if self.use_visual_obs == True:
            # if len(self.path) == 0:
            #     self.vae = AutoencoderPN(3, 64, 256).to(self.device)
            # else:
            #     self.vae = AutoencoderPN(3, 64, 256, path=self.path).to(self.device)
            if len(self.path) == 0:
                self.vae = EncoderDecoder(key_w=args).to(self.device)
            else:
                self.vae = EncoderDecoder(key_w=args).to(self.device)

        self.AC = ActorCritic(args).to(self.device)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_AC = torch.optim.Adam(({'params': self.AC.parameters()}, \
                                                  {'params': self.vae.parameters()}), lr=self.lr_ac, eps=1e-5)
        else:
            self.optimizer_AC = torch.optim.Adam(({'params': self.AC.parameters()}, \
                                                  {'params': self.vae.parameters()}), lr=self.lr_ac)

        # print(self)

    def evaluate(self, s):  # When evaluating the policy, we only use the mean
        if isinstance(s, dict):
            s_state = torch.tensor(s['state'], dtype=torch.float).to(self.device)
            s_oracle = torch.tensor(s['oracle_state'], dtype=torch.float).to(self.device)
            # s_camera_pc = torch.tensor(s['relocate-depth'], dtype=torch.float).to(self.device)

            depth_map = s['relocate-depth'].squeeze(2)
            mask = (s['relocate-segmentation'][..., 0] == 6)
            depth_map = np.multiply(depth_map, mask)

            # 将深度图转换为点云
            point_cloud = depth_to_point_cloud(depth_map, self.camera_intrinsic)
            point_cloud_new = point_cloud

            point_cloud = point_cloud_new.transpose(1, 0)
            true_index = np.nonzero(np.any(point_cloud, axis=1))[0]
            point_cloud = point_cloud[true_index]
            point_cloud = point_cloud[:3500, :]
            if len(point_cloud) < 3500:
                point_cloud = np.concatenate((point_cloud, np.zeros((3500 - len(point_cloud), 3))), axis=0)
            s_camera_pc = torch.tensor(point_cloud,dtype=torch.float32).to(self.device)

            encoding,coarse, restoration,_ = self.vae(s_camera_pc.unsqueeze(0))
            s = torch.concat((encoding.squeeze(0)*self.state_coef, s_state, s_oracle), dim=0)

            s = torch.unsqueeze(s, 0).to(self.device)
        else:
            s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)

        # s = torch.unsqueeze(torch.tensor(s, dtype=torch.float).to(self.device), 0)
        a, _ = self.AC(s)

        return a.detach().cpu().numpy().flatten()

    def choose_action(self, s):

        if isinstance(s, dict):
            s_state = torch.tensor(s['state'], dtype=torch.float).to(self.device)
            s_oracle = torch.tensor(s['oracle_state'], dtype=torch.float).to(self.device)
            depth_map = s['relocate-depth'].squeeze(2)
            mask = (s['relocate-segmentation'][..., 0] == 6)
            # print(mask.shape)
            # print(depth_map.shape)
            depth_map = np.multiply(depth_map, mask)

            # 将深度图转换为点云
            point_cloud = depth_to_point_cloud(depth_map, self.camera_intrinsic)
            point_cloud_new = point_cloud

            point_cloud = point_cloud_new.transpose(1, 0)
            true_index = np.nonzero(np.any(point_cloud, axis=1))[0]
            point_cloud = point_cloud[true_index]
            point_cloud = point_cloud[:3500, :]
            if len(point_cloud) < 3500:
                point_cloud = np.concatenate((point_cloud, np.zeros((3500 - len(point_cloud), 3))), axis=0)
            s_camera_pc = torch.tensor(point_cloud,dtype=torch.float32).to(self.device)

            encoding, coarse,restoration,_ = self.vae(s_camera_pc.unsqueeze(0))
            encoding=encoding*self.state_coef
            s = torch.concat((encoding.squeeze(0), s_state, s_oracle), dim=0)

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
            for index, state in enumerate(s):
                s_state = torch.tensor(state['state'], dtype=torch.float).to(self.device)
                s_oracle = torch.tensor(state['oracle_state'], dtype=torch.float).to(self.device)
                depth_map = state['relocate-depth'].squeeze(2)
                mask = (state['relocate-segmentation'][..., 0] == 6)
                depth_map = np.multiply(depth_map, mask)

                # 将深度图转换为点云
                point_cloud = depth_to_point_cloud(depth_map, self.camera_intrinsic)
                point_cloud_new = point_cloud

                point_cloud = point_cloud_new.transpose(1, 0)
                true_index = np.nonzero(np.any(point_cloud, axis=1))[0]
                point_cloud = point_cloud[true_index]
                point_cloud = point_cloud[:3500, :]
                if len(point_cloud) < 3500:
                    point_cloud = np.concatenate((point_cloud, np.zeros((3500 - len(point_cloud), 3))), axis=0)
                s_camera_pc = torch.tensor(point_cloud,dtype=torch.float32).to(self.device)

                encoding, coarse,restoration,_ = self.vae(s_camera_pc.unsqueeze(0))
                state_f = torch.concat((encoding.squeeze(0)*self.state_coef, s_state, s_oracle), dim=0)
                state_f = torch.unsqueeze(state_f, 0).to(self.device)

                if index == 0:
                    state_f_all = state_f

                else:

                    state_f_all = torch.concat([state_f_all, state_f], dim=0)


        if isinstance(s_, list):
            for index, state in enumerate(s_):
                s_state = torch.tensor(state['state'], dtype=torch.float).to(self.device)
                s_oracle = torch.tensor(state['oracle_state'], dtype=torch.float).to(self.device)
                depth_map = state['relocate-depth'].squeeze(2)
                mask = (state['relocate-segmentation'][..., 0] == 6)
                depth_map = np.multiply(depth_map, mask)

                # 将深度图转换为点云
                point_cloud = depth_to_point_cloud(depth_map, self.camera_intrinsic)
                point_cloud_new = point_cloud

                point_cloud = point_cloud_new.transpose(1, 0)
                true_index = np.nonzero(np.any(point_cloud, axis=1))[0]
                point_cloud = point_cloud[true_index]
                point_cloud = point_cloud[:3500, :]
                if len(point_cloud) < 3500:
                    point_cloud = np.concatenate((point_cloud, np.zeros((3500 - len(point_cloud), 3))), axis=0)
                s_camera_pc = torch.tensor(point_cloud,dtype=torch.float32).to(self.device)
                encoding, coarse,restoration,_ = self.vae(s_camera_pc.unsqueeze(0))
                state_f = torch.concat((encoding.squeeze(0)*self.state_coef, s_state, s_oracle), dim=0)

                state_f = torch.unsqueeze(state_f, 0).to(self.device)
                if index == 0:
                    state_f_all_ = state_f
                else:
                    state_f_all_ = torch.concat([state_f_all_, state_f], dim=0)

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
                for order, num in enumerate(index):
                    state = s[num]
                    s_state = torch.tensor(state['state'], dtype=torch.float).to(self.device)
                    s_oracle = torch.tensor(state['oracle_state'], dtype=torch.float).to(self.device)
                    depth_map = state['relocate-depth'].squeeze(2)
                    mask = (state['relocate-segmentation'][..., 0] == 6)
                    depth_map = np.multiply(depth_map, mask)

                    # 将深度图转换为点云
                    point_cloud = depth_to_point_cloud(depth_map, self.camera_intrinsic)
                    point_cloud_new = point_cloud

                    point_cloud = point_cloud_new.transpose(1, 0)
                    true_index = np.nonzero(np.any(point_cloud, axis=1))[0]
                    point_cloud = point_cloud[true_index]

                    """
                    cut point cloud
                    """
                    num_point = point_cloud.shape[0]
                    # print(num_point)
                    if num_point < 1200:
                        # del env
                        pass
                    else:
                        point_list = np.linspace(0, num_point - 1, num_point, dtype=int)
                        point_list = np.random.choice(point_list, 1200)
                        point_cloud = point_cloud[point_list]


                    s_camera_pc = torch.tensor(point_cloud,dtype=torch.float32).to(self.device).unsqueeze(0)
                    encoding, coarse,restoration,_ = self.vae(s_camera_pc)
                    state_f = torch.concat((encoding.squeeze(0)*self.state_coef, s_state, s_oracle), dim=0)
                    state_f = torch.unsqueeze(state_f, 0).to(self.device)
                    if order == 0:
                        state_f_all = state_f
                        pc_origi = s_camera_pc
                        pc_recon = restoration
                        pc_recon_coarse = coarse
                        encoding=encoding
                    else:
                        state_f_all = torch.concat([state_f_all, state_f], dim=0)
                        encoding=torch.concat([encoding,encoding],dim=0)
                        pc_origi = torch.concat([pc_origi, s_camera_pc], dim=0)
                        pc_recon = torch.concat([pc_recon, restoration], dim=0)
                        pc_recon_coarse = torch.concat([pc_recon_coarse, coarse], dim=0)
                dist_now = self.AC.get_dist(state_f_all)
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index])
                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1,
                                                                                             keepdim=True))  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]

                _, v_s = self.AC(state_f_all)
                critic_loss = F.mse_loss(v_target[index], v_s)
                for i in [pc_recon_coarse,pc_recon,pc_origi]:
                    print(i.shape)
                chamfer_l = ChamLoss(pc_recon_coarse,pc_recon,pc_origi,0.3)
                loss = -torch.min(surr1, surr2) + self.critic_coef * critic_loss - \
                       self.entropy_coef * dist_entropy + self.cham_coef * chamfer_l  # Trick 5: policy entropy
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

        torch.save(self.state_dict(), directory + '/AC_vae.pth')


if __name__ == '__main__':
    pass
