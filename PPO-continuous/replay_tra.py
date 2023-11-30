import time
import torch
import sys
import numpy as np
import os
from policy import TanhGaussianPolicy
from env_wrappers import NormalizedBoxEnv
import gym
import pytorch_util as ptu
from hand_teleop.env.rl_env.inspire_relocate_env import InspireRelocateRLEnv
from hand_teleop.env.rl_env.inspire_relocate_env_new_reward import InspireRelocateRLEnv as InspireRelocateRLEnvReward

# -------parameters-------
render = True
directory = './EX2333/best'
# directory = './EX_new_reward/SAC_2333/final'

ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)

env = InspireRelocateRLEnv(use_gui=True, robot_name="inspire_hand_free",
                           object_name="mustard_bottle", frame_skip=10, use_visual_obs=False)

# env = InspireRelocateRLEnvReward(use_gui=True, robot_name="inspire_hand_free",
#                     object_name="mustard_bottle", frame_skip=10, use_visual_obs=False)


obs_dim = env.observation_space.low.size
# action_dim = env.action_space.low.size
action_dim = env.real_dof
M = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
policy = TanhGaussianPolicy(
    obs_dim=obs_dim,
    action_dim=action_dim,
    hidden_sizes=[M, M],
).to(device)

# policy.load_state_dict(torch.load(os.path.join(directory, 'policy.pth'), map_location ='cpu'))
policy.load_state_dict(torch.load(os.path.join(directory, 'policy.pth')))
env.seed(989)
state = env.reset()

from sapien.utils import Viewer

base_env = env
viewer = Viewer(base_env.renderer)
viewer.set_scene(base_env.scene)
base_env.viewer = viewer
base_env.viewer.set_camera_xyz(x=-1, y=0, z=1)
base_env.viewer.set_camera_rpy(r=0, p=-np.arctan2(4, 2), y=0)

env.render()

step = 0
eps_reward = 0
t = 0

while True:
    t += 1
    print(t)
    state = np.array(state)
    a = policy.get_evaluate_action(state)
    a_apply = np.zeros(env.robot.dof)
    a_apply[env.total_activate_joint_index] = a.detach().cpu().numpy()

    next_state, r, terminal, _ = env.step(a_apply)
    time.sleep(1 / 10)
    if render:
        env.render()
    print(r)
    eps_reward += r
    step += 1
    state = next_state
    if terminal:
        break