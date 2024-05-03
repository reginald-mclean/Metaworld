import mujoco
import numpy as np
from gymnasium.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerDrawerCloseEnvV2(SawyerXYZEnv):
    _TARGET_RADIUS = 0.04

    def __init__(self, render_mode=None, reward_func_version="v2"):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.9, 0.0)
        obj_high = (0.1, 0.9, 0.0)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
        )

        self.reward_func_version = reward_func_version

        self.init_config = {
            "obj_init_angle": np.array(
                [
                    0.3,
                ],
                dtype=np.float32,
            ),
            "obj_init_pos": np.array([0.0, 0.9, 0.0], dtype=np.float32),
            "hand_init_pos": np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.maxDist = 0.15
        self.target_reward = 1000 * self.maxDist + 1000 * 2

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_drawer.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            target_to_obj,
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(target_to_obj <= (self.TARGET_RADIUS + 0.015)),
        }

        return reward, info

    def _get_pos_objects(self):
        return self.get_body_com("drawer_link") + np.array([0.0, -0.16, 0.05])

    def _get_quat_objects(self):
        return np.zeros(4)

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()

        # Compute nightstand position
        self.obj_init_pos = self._get_state_rand_vec()
        # Set mujoco body to computed position

        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drawer")
        ] = self.obj_init_pos
        # Set _target_pos to current drawer position (closed)
        self._target_pos = self.obj_init_pos + np.array([0.0, -0.16, 0.09])
        # Pull drawer out all the way and mark its starting position
        self._set_obj_xyz(-self.maxDist)
        self.obj_init_pos = self._get_pos_objects()

        return self._get_obs()

    def compute_reward(self, action, obs):
        if self.reward_func_version == "v2":
            obj = obs[4:7]
            tcp = self.tcp_center
            target = self._target_pos.copy()

            target_to_obj = obj - target
            target_to_obj = np.linalg.norm(target_to_obj)
            target_to_obj_init = self.obj_init_pos - target
            target_to_obj_init = np.linalg.norm(target_to_obj_init)

            in_place = reward_utils.tolerance(
                target_to_obj,
                bounds=(0, self.TARGET_RADIUS),
                margin=abs(target_to_obj_init - self.TARGET_RADIUS),
                sigmoid="long_tail",
            )

            handle_reach_radius = 0.005
            tcp_to_obj = np.linalg.norm(obj - tcp)
            tcp_to_obj_init = np.linalg.norm(self.obj_init_pos - self.init_tcp)
            reach = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, handle_reach_radius),
                margin=abs(tcp_to_obj_init - handle_reach_radius),
                sigmoid="gaussian",
            )
            gripper_closed = min(max(0, action[-1]), 1)

            reach = reward_utils.hamacher_product(reach, gripper_closed)
            tcp_opened = 0
            object_grasped = reach

            reward = reward_utils.hamacher_product(reach, in_place)
            if target_to_obj <= self.TARGET_RADIUS + 0.015:
                reward = 1.0

            reward *= 10

            return (reward, target_to_obj)
        elif self.reward_func_version == 'v1':
            objPos = obs[4:7]
            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            pullGoal = self._target_pos[1]

            reachDist = np.linalg.norm(objPos - fingerCOM)

            pullDist = np.abs(objPos[1] - pullGoal)

            c1 = 1000
            c2 = 0.01
            c3 = 0.001

            if reachDist < 0.05:
                pullRew = 1000 * (self.maxDist - pullDist) + c1 * (
                    np.exp(-(pullDist**2) / c2) + np.exp(-(pullDist**2) / c3)
                )
                pullRew = max(pullRew, 0)
            else:
                pullRew = 0

            reward = -reachDist + pullRew

            return [reward, pullDist]
        elif self.reward_func_version == 'text2reward':
            # Distance between robot's gripper and the drawer's handle
            gripper_handle_dist = np.linalg.norm(obs[:3] - obs[4:7])

            # Difference between current state of drawer's position and its goal state
            drawer_goal_dist = np.linalg.norm(obs[4:7] - obs[-3:])

            # The gripper should be able to grasp the handle. We can encourage this by
            # adding a reward component that takes into account the gripper's openness.
            # Assuming that a gripper_openness of -1 means fully open, and 1 means fully closed.
            gripper_reward = -abs(obs[3] + 1)  # Maximum reward when fully open.

            # Regularization of the robot's action: we want the robot to perform the task
            # with minimal and efficient action. A common way to achieve this is to penalize
            # large actions.
            action_reg = -np.linalg.norm(action)

            # Weights for each reward component
            w1, w2, w3, w4 = 1.0, 1.0, 0.5, 0.1

            # Combine the reward components
            reward = w1 * (-gripper_handle_dist) + w2 * (-drawer_goal_dist) + w3 * gripper_reward + w4 * action_reg

            '''# Constants for reward tuning
            distance_weight = -1.0  # Negative because we want to minimize distance
            closure_weight = 1.0  # Positive as we want to maximize the state of closure
            action_penalty = -0.1  # Penalize the magnitude of action to ensure smoothness
    
            # Calculate distance from end-effector to the handle of the drawer (assuming obj1 is the drawer)
            handle_position = obs[4:7]  # Assuming the handle's position is synonymous with the object's position
            distance_to_handle = np.linalg.norm(obs[:3] - handle_position)
    
            # Calculate how closed the drawer is (simplified assumption: closer to goal_position means more closed)
            # We assume goal_position indicates the fully closed position of the drawer
            drawer_closure = -np.linalg.norm(obs[4:7] - obs[-3:])  # More negative when farther from goal
    
            # Calculate action penalty (sum of squared action values)
            action_cost = np.sum(np.square(action))
    
            # Combine the parts of the reward function
            reward = (distance_weight * distance_to_handle +
                closure_weight * drawer_closure +
                action_penalty * action_cost)'''
    
            obj = obs[4:7]
            tcp = self.tcp_center
            target = self._target_pos.copy()

            target_to_obj = obj - target
            target_to_obj = np.linalg.norm(target_to_obj)
            return reward, target_to_obj
        elif self.reward_func_version =='t2r3':
            distance_to_handle = np.linalg.norm(obs[:3] - obs[4:7])

            # Calculate the current drawer position's distance to the goal position
            drawer_to_goal_distance = np.linalg.norm(obs[4:7] - obs[-3:])

            # Normalize distances for reward scaling
            # Assuming max possible distance in the workspace for normalization (workspace_diagonal)
            workspace_diagonal = 2.0  # Example value, adjust according to actual workspace dimensions
            normalized_dist_to_handle = distance_to_handle / workspace_diagonal
            normalized_drawer_to_goal_distance = drawer_to_goal_distance / workspace_diagonal

            # Invert distances to make rewards positive (closer to zero means better, so subtract from 1)
            reward_dist_to_handle = 1 - normalized_dist_to_handle
            reward_drawer_to_goal = 1 - normalized_drawer_to_goal_distance

            # Weight components of the reward function
            # We might prioritize reaching the handle first and then focus on closing the drawer
            weight_handle = 0.6
            weight_drawer = 0.4

            # Combine the rewards
            total_reward = weight_handle * reward_dist_to_handle + weight_drawer * reward_drawer_to_goal

            obj = obs[4:7]
            tcp = self.tcp_center
            target = self._target_pos.copy()

            target_to_obj = obj - target
            target_to_obj = np.linalg.norm(target_to_obj)
            return total_reward, target_to_obj
