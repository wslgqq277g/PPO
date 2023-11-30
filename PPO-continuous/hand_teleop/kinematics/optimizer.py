from typing import List

import nlopt
import numpy as np
import sapien.core as sapien
import torch


class Optimizer:
    def __init__(self, robot: sapien.Articulation, robot_name, target_joint_names: List[str]):
        self.robot = robot
        self.robot_name = robot_name
        self.model = robot.create_pinocchio_model()

        joint_names = [joint.get_name() for joint in robot.get_active_joints()]
        target_joint_index = []
        for target_joint_name in target_joint_names:
            if target_joint_name not in joint_names:
                raise ValueError(f"Joint {target_joint_name} given does not appear to be in robot XML.")
            target_joint_index.append(joint_names.index(target_joint_name))
        self.target_joint_indices = np.array(target_joint_index)
        
        self.target_actiavte_joint_indices = self.target_joint_indices[:-6]
        self.target_coupled_joint_indices = self.target_joint_indices[-6:]

        self.fixed_joint_indices = np.array([i for i in range(robot.dof) if i not in target_joint_index], dtype=int)
        self.opt = nlopt.opt(nlopt.LD_LBFGS, len(target_joint_index))
        self.dof = len(target_joint_index)

    def set_joint_limit(self, joint_limits: np.ndarray):
        if joint_limits.shape != (self.dof, 2):
            raise ValueError(f"Expect joint limits have shape: {(self.dof, 2)}, but get {joint_limits.shape}")
        self.opt.set_lower_bounds(joint_limits[:, 0].tolist())
        self.opt.set_upper_bounds(joint_limits[:, 1].tolist())


    def get_last_result(self):
        return self.opt.last_optimize_result()

    def get_link_names(self):
        return [link.get_name() for link in self.robot.get_links()]


class PositionOptimizer(Optimizer):
    def __init__(self, robot: sapien.Articulation, robot_name, target_joint_names: List[str], target_link_names: List[str],
                 huber_delta=0.01, norm_delta=4e-3):
        super().__init__(robot, robot_name, target_joint_names)
        self.body_names = target_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta)
        self.norm_delta = norm_delta

        # Sanity check
        target_link_index = []
        # print(target_link_names)
        for target_link_name in target_link_names:
            if target_link_name not in self.get_link_names():
                raise ValueError(f"Body {target_link_name} given does not appear to be in robot XML.")
            target_link_index.append(self.get_link_names().index(target_link_name))
        self.target_link_indices = target_link_index
        # print(target_link_index)
        # assert 0
        # Use local jacobian if target link name <= 2, otherwise first cache all jacobian and then get then all
        # This is only for the speed but will not affect the performance
        if len(target_link_names) <= 40:
            self.use_sparse_jacobian = True
        else:
            self.use_sparse_jacobian = False

        if "inspire" in self.robot_name:
            # ef_args = [-0.12445664, -0.24671012, -0.40052218,  0.61364766,  0.02506414,]
            # ab_args = [-0.19567243,  0.38251126, -0.5014147,  -0.90547973,  0.001485, ]
            # ac_args = [0.00232606,  0.02423625,  0.62661481, -1.31523101,  0.02723364,]

            # ef_args = [-0.12445664, -0.24605431, -0.39954848,  0.61470162,  0.39600216]
            # ab_args = [ 0.19567243, -0.00527726,  0.22106028,  1.20987692, -0.01086015]
            # ac_args = [-0.00232606, -0.02872063, -0.66490035,  0.69327858, -0.27172935]


            ef_args = [-0.12445664, -0.24605431, -0.39954848,  0.61470162,  0.39600216]
            ab_args = [-0.19567243,  0.00527726, -0.22106028, -1.20987692, -0.27953349]
            ac_args = [ 0.00232606,  0.02872063,  0.66490035, -0.69327858, -0.64337397]
            
            self.poly_ef = np.poly1d(ef_args)
            self.poly_ab = np.poly1d(ab_args)
            self.poly_ac = np.poly1d(ac_args)

            self.d_poly_ef = self.poly_ef.deriv()
            self.d_poly_ab = self.poly_ab.deriv()
            self.d_poly_ac = self.poly_ac.deriv()

    def _get_objective_function(self, target_pos: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        qpos = np.zeros(self.robot.dof)
        qpos[self.fixed_joint_indices] = fixed_qpos

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            if "inspire" in self.robot_name:
                # self.target_actiavte_joint_indices 
                # self.target_coupled_joint_indices 
                # ---- fo inspire hand------
                x_freebase = x[:6]
                x_activate = x[6:-6]

                Joint_E = x_activate[:4]
                Joint_F = self.poly_ef(Joint_E)

                Joint_A = x_activate[5]
                Joint_B = self.poly_ab(Joint_A)
                Joint_C = self.poly_ac(Joint_A)

                x_coupled = np.concatenate((Joint_F, np.array([Joint_B]), np.array([Joint_C])))

                x = np.concatenate((x_freebase, x_activate, x_coupled))


            qpos[self.target_joint_indices] = x

            # print('--------new--------')
            # print('qpos:', qpos)
            self.model.compute_forward_kinematics(qpos)
            target_link_poses = [self.model.get_link_pose(index) for index in self.target_link_indices]
            # print("link indices", self.target_link_indices)
            # print(target_link_poses)
            # for pose in target_link_poses:
            #     print(pose.to_transformation_matrix())
            body_pos = np.array([pose.p for pose in target_link_poses])
            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos) # you get the body pos and then need grad
            torch_body_pos.requires_grad_()
            torch_target_pos = torch.as_tensor(target_pos)
            torch_target_pos.requires_grad_(False)

            # Loss term for kinematics retargeting based on 3D position error
            huber_distance = self.huber_loss(torch_body_pos, torch_target_pos)
            # huber_distance = torch.norm(torch_body_pos - torch_target_pos, dim=1).mean()
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                if self.use_sparse_jacobian:
                    # (6, 3, 18)
                    jacobians = []
                    for i, index in enumerate(self.target_link_indices):
                        link_spatial_jacobian = self.model.compute_single_link_local_jacobian(qpos, index)[:3,
                                                self.target_joint_indices]
                        link_rot = self.model.get_link_pose(index).to_transformation_matrix()[:3, :3]
                        link_kinematics_jacobian = link_rot @ link_spatial_jacobian
                        # （3, 18）
                        if "inspire" in self.robot_name:

                            # [0-5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17]
                            # base, E, E, E, E, D, A,  F, F, F, F, B, C

                            Jacobian_E = link_kinematics_jacobian[:, 6:10]
                            Jacobian_A = link_kinematics_jacobian[:, 11]
                            Jacobian_F = link_kinematics_jacobian[:, 12:16]
                            Jacobian_B = link_kinematics_jacobian[:, 16]
                            Jacobian_C = link_kinematics_jacobian[:, 17]

                            Joint_E = qpos[6:10]
                            Joint_A = qpos[11]

                            dFdE = self.d_poly_ef(Joint_E)
                            dBdA = self.d_poly_ab(Joint_A)
                            dCdA = self.d_poly_ac(Joint_A)

                            Jacobian_E_new = Jacobian_E + Jacobian_F * dFdE
                            Jacobian_A_new = Jacobian_A + Jacobian_B * dBdA + Jacobian_C * dCdA 
                            
                            # print('------------------')
                            # print(Joint_E)
                            # print("JF:", Jacobian_F)
                            # print('dFdE', dFdE)
                            # print("JF after:", aaa)
                            # print(dFdE)
                            # print(Jacobian_E)
                            # print(aaa)

                            # coupled joint
                            link_kinematics_jacobian[:, 12:] = 0
                            # E
                            link_kinematics_jacobian[:, 6:10] = Jacobian_E_new
                            # A
                            link_kinematics_jacobian[:, 11] = Jacobian_A_new

                        jacobians.append(link_kinematics_jacobian)
                    jacobians = np.stack(jacobians, axis=0)
                else:
                    self.model.compute_full_jacobian(qpos)
                    jacobians = [self.model.get_link_jacobian(index, local=True)[:3, self.target_joint_indices] for
                                 index in self.target_link_indices]

                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]
                grad_qpos = np.matmul(grad_pos, np.array(jacobians)) # according to the chain rule
                grad_qpos = grad_qpos.mean(1).sum(0)

                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective

    def retarget(self, target_pos, fixed_qpos, last_qpos=None, verbose=True):
        if len(fixed_qpos) != len(self.fixed_joint_indices):
            raise ValueError(
                f"Optimizer has {len(self.fixed_joint_indices)} joints but non_target_qpos {fixed_qpos} is given")
        if last_qpos is None:
            last_qpos = np.zeros(self.dof)
        if isinstance(last_qpos, np.ndarray):
            last_qpos = last_qpos.astype(np.float32)
        if "inspire" in self.robot_name:
            x_freebase = last_qpos[:6]
            x_activate = last_qpos[6:-6]

            Joint_E = x_activate[:4]
            Joint_F = self.poly_ef(Joint_E)
            Joint_A = x_activate[5]
            Joint_B = self.poly_ab(Joint_A)
            Joint_C = self.poly_ac(Joint_A)
            x_coupled = np.concatenate((Joint_F, np.array([Joint_B]), np.array([Joint_C])))
            last_qpos = np.concatenate((x_freebase, x_activate, x_coupled))


        last_qpos = list(last_qpos)
        # print(last_qpos)
        # print()


        objective_fn = self._get_objective_function(target_pos, fixed_qpos, np.array(last_qpos).astype(np.float32))
        self.opt.set_min_objective(objective_fn)
        self.opt.set_ftol_abs(1e-5)
        try:
            qpos = self.opt.optimize(last_qpos)
        except RuntimeError as e:
            print('??????????????')
            print(e)
            return np.array(last_qpos)
        min_value = self.opt.last_optimum_value()
        if verbose:
            print(f"Last distance: {min_value}")
        return qpos


def test_position_optimizer():
    import time
    from hand_teleop.utils.common_robot_utils import load_robot

    np.set_printoptions(precision=4)
    np.random.seed(1)

    # SAPIEN Scene
    engine = sapien.Engine()
    scene = engine.create_scene()
    scene.set_timestep(1 / 125)

    # Articulation
    robot = load_robot(scene, "adroit_hand_free")
    scene.step()

    # Optimizer
    joint_names = [joint.get_name() for joint in robot.get_active_joints()][6:]
    link_names = ["thtip", "fftip", "mftip", "rftip", "lftip"]
    joint_limit = robot.get_qlimits()
    optimizer = PositionOptimizer(robot, joint_names, link_names)
    optimizer.set_joint_limit(joint_limit[6:])

    random_qpos = np.random.rand(robot.dof)
    random_qpos = random_qpos.clip(joint_limit[:, 0], joint_limit[:, 1])
    robot.set_qpos(random_qpos)
    random_target = np.array([robot.get_links()[i].get_pose().p for i in optimizer.target_link_indices])
    tic = time.time()
    init_qpos = np.clip(random_qpos + np.random.randn(robot.dof) * 0.1, joint_limit[:, 0], joint_limit[:, 1])
    computed_qpos = optimizer.retarget(random_target, last_qpos=init_qpos[6:], fixed_qpos=random_qpos[:6])
    print(f"Kinematics Retargeting computation takes {time.time() - tic}s")
    print(np.mean(np.abs(computed_qpos - random_qpos[6:])))

    qpos = np.copy(random_qpos)
    qpos[6:] = computed_qpos
    robot.set_qpos(qpos)
    computed_target = np.array([robot.get_links()[i].get_pose().p for i in optimizer.target_link_indices])
    print("distance from target", np.mean(np.linalg.norm(computed_target - random_target, axis=1)))
    robot.set_qpos(init_qpos)
    init_target = np.array([robot.get_links()[i].get_pose().p for i in optimizer.target_link_indices])
    print("distance from init", np.mean(np.linalg.norm(computed_target - init_target, axis=1)))


if __name__ == '__main__':
    test_position_optimizer()
