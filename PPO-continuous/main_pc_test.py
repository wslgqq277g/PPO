import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym

import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer_PC as ReplayBuffer
from ppo_continuous_pc import PPO_continuous_pc as PPO_continuous
import os
from hand_teleop.env.rl_env.inspire_relocate_pc_env import InspireRelocateRLEnv
import sys
# sys.path.append('../..')
import open3d as o3d
import sapien.core as sapien

sys.path.append(os.path.join(os.getcwd(),'..'))
from dexpoint.real_world import task_setting


def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            a = action
            # #print(a.shape,'a')
            a_apply = np.zeros(env.robot.dof)
            a_apply[env.total_activate_joint_index] = a

            # s_, r, done, _ = env.step(action)
            s_, r, done, _ = env.step(a_apply)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times


def main(args, seed):

    # env = gym.make(env_name)
    # env_evaluate = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    env = InspireRelocateRLEnv( rotation_reward_weight=0,
                               robot_name="inspire_hand_free",
                               object_name="mustard_bottle",
                                use_gui=False,
                                frame_skip=10,
                                use_visual_obs=True,
                                no_rgb=True)
    env_evaluate = InspireRelocateRLEnv( rotation_reward_weight=0,
                                        robot_name="inspire_hand_free",
                                        object_name="mustard_bottle", frame_skip=10, use_visual_obs=True,no_rgb=True)
    #use_visual_obs=True--->self.get_observation = self.get_visual_observation
    # def get_visual_observation(self):
    #     camera_obs = self.get_camera_obs()
    #     robot_obs = self.get_robot_state()
    #     oracle_obs = self.get_oracle_state()
    #     camera_obs.update(dict(state=robot_obs, oracle_state=oracle_obs))
    #     return camera_obs
    #obs--->get_visual_observation
    env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # args.state_dim = env.observation_space.shape[0]

    # #print(type(env.observation_space) is gym.spaces.Dict,'kk222222222kk')

    # #print(env.observation_space.spaces['state'].shape,'kk444444444kk')
    # assert False
    args.action_dim = env.real_dof # 12
    args.max_action = float(env.action_space.high[0])
    # args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    args.max_episode_steps = 300000
    #print("env={}".format(env_name))
    # #print("state_dim={}".format(args.state_dim))
    #print("action_dim={}".format(args.action_dim))
    #print("max_action={}".format(args.max_action))
    #print("max_episode_steps={}".format(args.max_episode_steps))

    env.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate"])
    # Specify observation modality
    env.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate_noise"])

    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    env_evaluate.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate"])
    # Specify observation modality
    env_evaluate.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate_noise"])
    args.state_dim = env.observation_space['state'].shape[0]+env.observation_space['oracle_state'].shape[0]+\
        256
    save_dir = os.path.join(os.getcwd(), '{}'.format(seed)+'pc')
    obs = env.reset()
    #print("For state task, observation is a numpy array. For visual tasks, observation is a python dict.")

    print("Observation keys")
    print(obs.keys())

    isExists = os.path.exists(save_dir)
    if isExists == False:
        os.mkdir(save_dir)

    evaluate_num = 0  # Record the number of evaluations
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)

    # Build a tensorboard
    writer = SummaryWriter(log_dir=save_dir)
    print(save_dir, ':save_dir')
    # state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
    base_env = env
    # from sapien.utils import Viewer

    # viewer = Viewer(base_env.renderer)
    # viewer.set_scene(base_env.scene)
    # base_env.viewer = viewer
    # base_env.viewer.set_camera_xyz(x=-1, y=0, z=1)
    # base_env.viewer.set_camera_rpy(r=0, p=-np.arctan2(4, 2), y=0)
    # env.render()
    while total_steps < args.max_train_steps:
        best_reward = -9999999999999999999
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False

        '''visualize pc'''
        if args.visualize_pc:# while not done:
            pc = s["relocate-point_cloud"]
            cloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc))
            coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([cloud, coordinate])


        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            a = action
            a_apply = np.zeros(env.robot.dof)
            a_apply[env.total_activate_joint_index] = a

            s_, r, done, _ = env.step(a_apply)


            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)
            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False
            # Take the 'action'，but store the original 'a'（especially for Beta）
            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1

            # env.scene.step()
            # env.scene.update_render()
            # env.render()

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                # evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                evaluate_reward = evaluate_policy(args, env_evaluate, agent, None)

                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                writer.add_scalar('step_rewards', evaluate_reward, global_step=total_steps)
                if evaluate_reward >= best_reward:
                    print("evaluate_num:{} \t evaluate_best_reward:{} \t".format(evaluate_num, evaluate_reward))
                    agent.save(os.path.join(save_dir, 'best'))
                    best_reward = evaluate_reward

                """save pc"""
                # pc = obs["relocate-point_cloud"]
                # The name of the key in observation is "CAMERA_NAME"-"MODALITY_NAME".
                # While CAMERA_NAME is defined in task_setting.CAMERA_CONFIG["relocate"], name is point_cloud.
                # See example_use_multi_camera_visual_env.py for more modalities.

                # simulation_steps = rl_steps * env.frame_skip
                # print(f"Single process for point-cloud environment with {rl_steps} RL steps "
                #       f"(= {simulation_steps} simulation steps) takes {elapsed_time}s.")
                # print(
                #     "Keep in mind that using multiple processes during RL training can significantly increase the speed.")
                # env.scene = None

                # Note1: point cloud are represented x-forward convention in robot frame, see visualization to get a sense of frame
                # Note2: you may also need to remove the points with smaller depth
                # Note3: as described in the paper, the point are cropped into the robot workspace without background and table
                # cloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc))
                # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
                # o3d.visualization.draw_geometries([cloud, coordinate])

            if total_steps % args.save_freq == 0:
                agent.save(os.path.join(save_dir, 'final'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--log_dir", type=str, default='./Ex', help=" The log directory")
    parser.add_argument("--seed", type=int, default=2333, help=" Seed")

    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=1000, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")


    #shareac
    parser.add_argument("--lr_ac", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--critic_coef", type=float, default=0.5, help="critic coef")

    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=False, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")

    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--use_visual_obs", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--path", type=str, default='', help="Trick 10: tanh activation function")
    parser.add_argument("--visualize_pc", type=str, default='', help="Trick 10: tanh activation function")


    args = parser.parse_args()

    main(args, seed=args.seed)
