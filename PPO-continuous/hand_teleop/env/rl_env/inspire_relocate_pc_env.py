from functools import cached_property
from typing import Optional
from hand_teleop.env.rl_env.point_net import PointNet

import numpy as np
import sapien.core as sapien
import transforms3d
from sapien.utils import Viewer
# import sys
# sys.path.append('../../')
from hand_teleop.env.rl_env.inspire_base import BaseRLEnv

from hand_teleop.env.sim_env.relocate_env import RelocateEnv
from hand_teleop.real_world import lab

OBJECT_LIFT_LOWER_LIMIT = -0.03


class InspireRelocateRLEnv(RelocateEnv, BaseRLEnv):
    def __init__(self, use_gui=False,frame_skip=5, robot_name="inspire_hand_free", constant_object_state=False,
                 rotation_reward_weight=0, object_category="YCB", object_name="tomato_soup_can", object_scale=1.0,
                 randomness_scale=1, friction=1, object_pose_noise=0.01, **renderer_kwargs):
        super().__init__(use_gui, frame_skip, object_category, object_name, object_scale, randomness_scale, friction,
                         **renderer_kwargs)
        self.setup(robot_name)

        self.constant_object_state = constant_object_state
        self.rotation_reward_weight = rotation_reward_weight
        self.object_pose_noise = object_pose_noise
        # print(self.robot.get_links(),'asdas')
        # assert False
        # Parse link name
        self.palm_link_name = self.robot_info.palm_name
        self.palm_link = [link for link in self.robot.get_links() if link.get_name() == self.palm_link_name][0]

        # Object init pose
        self.object_episode_init_pose = sapien.Pose()

        # real DOF
        self.real_dof = 12
    #PC env should not includ object's pose

    def get_oracle_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        # object_pose = self.object_episode_init_pose if self.constant_object_state else self.manipulated_object.get_pose()
        # object_pose_vec = np.concatenate([object_pose.p, object_pose.q])
        palm_pose = self.palm_link.get_pose()
        # target_in_object = self.target_pose.p - object_pose.p
        target_in_palm = self.target_pose.p - palm_pose.p
        # object_in_palm = object_pose.p - palm_pose.p
        palm_v = self.palm_link.get_velocity()
        palm_w = self.palm_link.get_angular_velocity()
        # theta = np.arccos(np.clip(np.power(np.sum(object_pose.q * self.target_pose.q), 2) * 2\
        #                           - 1, -1 + 1e-8, 1 - 1e-8))
        return np.concatenate(
            [robot_qpos_vec, palm_v, palm_w, target_in_palm,
             self.target_pose.q])

    def get_robot_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        palm_pose = self.palm_link.get_pose()
        return np.concatenate([robot_qpos_vec, palm_pose.p, self.target_pose.p, self.target_pose.q])

    def get_reward(self, action):
        object_pose = self.manipulated_object.get_pose()
        palm_pose = self.palm_link.get_pose()
        is_contact = self.check_contact(self.robot_collision_links, [self.manipulated_object])

        reward = -0.1 * min(np.linalg.norm(palm_pose.p - object_pose.p), 0.5)
        if is_contact:
            reward += 0.1
            lift = min(object_pose.p[2], self.target_pose.p[2]) - self.object_height
            lift = max(lift, 0)
            reward += 5 * lift
            if lift > 0.015:
                reward += 2
                obj_target_distance = min(np.linalg.norm(object_pose.p - self.target_pose.p), 0.5)
                reward += -1 * min(np.linalg.norm(palm_pose.p - self.target_pose.p), 0.5)
                reward += -3 * obj_target_distance  # make object go to target

                if obj_target_distance < 0.2:
                    reward += (0.1 - obj_target_distance) * 20
                    theta = np.arccos(
                        np.clip(np.power(np.sum(object_pose.q * self.target_pose.q), 2) * 2 - 1, -1 + 1e-8, 1 - 1e-8))
                    reward += max((np.pi / 2 - theta) * self.rotation_reward_weight, 0)
                    if theta < np.pi / 4 and self.rotation_reward_weight >= 1e-6:
                        reward += (np.pi / 4 - theta) * 6 * self.rotation_reward_weight

        return reward

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        super().reset(seed=seed)
        # if not self.is_robot_free:
        #     qpos = np.zeros(self.robot.dof)
        #     xarm_qpos = self.robot_info.arm_init_qpos
        #     qpos[:self.arm_dof] = xarm_qpos
        #     self.robot.set_qpos(qpos)
        #     self.robot.set_drive_target(qpos)
        #     init_pos = np.array(lab.ROBOT2BASE.p) + self.robot_info.root_offset
        #     init_pose = sapien.Pose(init_pos, transforms3d.euler.euler2quat(0, 0, 0))
        # else:
        # init_pose = sapien.Pose(np.array([-0.4, 0, 0.2]), transforms3d.euler.euler2quat(0, np.pi / 2, 0))
        init_pose = sapien.Pose(np.array([-0.4, 0, 0.15]), transforms3d.euler.euler2quat(np.pi/2, 0, np.pi/4))
        self.robot.set_pose(init_pose)
        self.reset_internal()
        # q_initial = np.array([ 0, 0, 0, 0, 0, 0,
        #                         0, 0.396, 0, 0.396, 0, 0.396,
        #                         0, 0.396, 0.36, -0.48, 0.2393, -0.16])

        q_initial = np.array([ 0, 0, 0, 0, 0, 0,
                                0, 0.396, 0, 0.396, 0, 0.396,
                                0, 0.396, -1.24, -0.48, 0.2393, -0.16])


        # [-1.63, -1.48, -1.24, -0.48, -0.54, -0.758, ]
        # [  0.0,   0.4,  0.36,  0.2,  0.241, -0.158,]
        # ['E',     'F',   'D',   'A',   'B',   'C']
        self.robot.set_qpos(q_initial)
        self.object_episode_init_pose = self.manipulated_object.get_pose()
        random_quat = transforms3d.euler.euler2quat(*(self.np_random.randn(3) * self.object_pose_noise * 10))
        random_pos = self.np_random.randn(3) * self.object_pose_noise
        self.object_episode_init_pose = self.object_episode_init_pose * sapien.Pose(random_pos, random_quat)
        # print(self.object_episode_init_pose)
        return self.get_observation()



        """in rl_env/inspire_base"""
        # if self.use_visual_obs:
        #     self.get_observation = self.get_visual_observation
        #     if not self.no_rgb:
        #         add_default_scene_light(self.scene, self.renderer)
        # else:
        #     self.get_observation = self.get_oracle_state



        # def get_visual_observation(self):
        #     camera_obs = self.get_camera_obs()
        #     robot_obs = self.get_robot_state()
        #     oracle_obs = self.get_oracle_state()
        #     camera_obs.update(dict(state=robot_obs, oracle_state=oracle_obs))
        #     return camera_obs

    @cached_property
    def obs_dim(self):
        if not self.use_visual_obs:
            print(self.robot.dof,'dof')
            return self.robot.dof + 7 + 6 + 9 + 4 + 1
        else:
            print(self.get_robot_state().shape,'robot_state')
            return len(self.get_robot_state())
    # def get_robot_state(self):
    #     robot_qpos_vec = self.robot.get_qpos()
    #     palm_pose = self.palm_link.get_pose()
    #     return np.concatenate([robot_qpos_vec, palm_pose.p, self.target_pose.p, self.target_pose.q])

    def is_done(self):
        # print(self.manipulated_object.pose.p[2]- self.object_height )
        # print(OBJECT_LIFT_LOWER_LIMIT)
        return self.manipulated_object.pose.p[2] - self.object_height < OBJECT_LIFT_LOWER_LIMIT

    @cached_property
    def horizon(self):
        return 250

import time
def main_env():
    env = InspireRelocateRLEnv(use_gui=True, robot_name="inspire_hand_free",
                        object_name="mustard_bottle", frame_skip=10, use_visual_obs=False)
    base_env = env
    robot_dof = env.robot.dof
    env.seed(0)
    env.reset()
    viewer = Viewer(base_env.renderer)
    viewer.set_scene(base_env.scene)
    base_env.viewer = viewer
    base_env.viewer.set_camera_xyz(x=-1, y=0, z=1)
    base_env.viewer.set_camera_rpy(r=0, p=-np.arctan2(4, 2), y=0)


    # for i in range(100):
    #     env.reset()
    #     env.render()
    #     time.sleep(1)

    # viewer.toggle_pause(True)
    for i in range(5000):
        # print(i)
        # print(robot_dof)
        action = np.zeros(robot_dof)
        # action = np.array([0, 0, 0, 0, 0, 0,
        #                    1, 0, 1, 0, 1, 0,
        #                    1, 0, 1, -1, 0, 0])

        action = np.array([0, 0, 0, 0, 0, 0,
                           1, 0, 1, 0, 1, 0,
                           1, 0, -1, -1, 0, 0])
        # action[15] = 1 - i*0.001

        # action[12] = -0.5
        # action[14] = -0.5
        # action[15] = -0.5
        # action[0] = 0.1
        # action[1] = -0.1
        # action[1] = 0.005
        # action[14] = 0.5
        # self.activate_joint_index = [6, 8, 10, 12, 14, 15]
        # self.coupled_joint_index = [7, 9, 11, 13, 16, 17]
        # action[1] =0.01
        # action[[6, 8, 10, 12]] = 0
        # action[14] = 0.1
        # action[15] = 0
        # action[12] = -1
        # assert 0
        # action[2] =0.01
        # action[1] = i * 0.02
        obs, reward, done, _ = env.step(action)
        print('obs:')
        print(obs[:robot_dof])

        env.render()


    while not viewer.closed:
        env.render()


if __name__ == '__main__':
    main_env()
