import mujoco
import numpy as np
from gymnasium.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerDrawerOpenEnvV2(SawyerXYZEnv):
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

        self.maxDist = 0.2
        self.target_reward = 1000 * self.maxDist + 1000 * 2

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_drawer.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (reward, handle_error) = self.compute_reward(action, obs)

        info = {
            "success": float(handle_error <= 0.03),
        }

        return reward, info

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id("objGeom")

    def _get_pos_objects(self):
        return self.get_body_com("drawer_link") + np.array([0.0, -0.16, 0.0])

    def _get_quat_objects(self):
        return self.data.body("drawer_link").xquat

    def reset_model(self):
        self._reset_hand()
        self.prev_obs = self._get_curr_obs_combined_no_goal()

        # Compute nightstand position
        self.obj_init_pos = self._get_state_rand_vec()
        # Set mujoco body to computed position
        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drawer")
        ] = self.obj_init_pos

        # Set _target_pos to current drawer position (closed) minus an offset
        self._target_pos = self.obj_init_pos + np.array(
            [0.0, -0.16 - self.maxDist, 0.09]
        )
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs()

    def compute_reward(self, action, obs):
        if self.reward_func_version == "v2":
            gripper = obs[:3]
            handle = obs[4:7]

            handle_error = np.linalg.norm(handle - self._target_pos)

            reward_for_opening = reward_utils.tolerance(
                handle_error, bounds=(0, 0.02), margin=self.maxDist, sigmoid="long_tail"
            )

            handle_pos_init = self._target_pos + np.array([0.0, self.maxDist, 0.0])
            # Emphasize XY error so that gripper is able to drop down and cage
            # handle without running into it. By doing this, we are assuming
            # that the reward in the Z direction is small enough that the agent
            # will be willing to explore raising a finger above the handle, hook it,
            # and drop back down to re-gain Z reward
            scale = np.array([3.0, 3.0, 1.0])
            gripper_error = (handle - gripper) * scale
            gripper_error_init = (handle_pos_init - self.init_tcp) * scale

            reward_for_caging = reward_utils.tolerance(
                np.linalg.norm(gripper_error),
                bounds=(0, 0.01),
                margin=np.linalg.norm(gripper_error_init),
                sigmoid="long_tail",
            )

            reward = reward_for_caging + reward_for_opening
            reward *= 5.0

            return (
                reward,
                handle_error,
            )
        elif self.reward_func_version == 'v1':
            del action

            objPos = obs[4:7]
            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2
            pullGoal = self._target_pos
            pullDist = np.abs(objPos[1] - pullGoal[1])
            reachDist = np.linalg.norm(objPos - fingerCOM)
            reachRew = -reachDist

            self.reachCompleted = reachDist < 0.05

            c1 = 1000
            c2 = 0.01
            c3 = 0.001

            if self.reachCompleted:
                pullRew = 1000 * (self.maxDist - pullDist) + c1 * (
                    np.exp(-(pullDist**2) / c2) + np.exp(-(pullDist**2) / c3)
                )
                pullRew = max(pullRew, 0)
            else:
                pullRew = 0

            reward = reachRew + pullRew

            return [reward, pullDist]
        elif self.reward_func_version == 'text2reward':
            # Define weights for different components of the reward
            w1, w2, w3, w4 = 1.0, 1.0, 0.01, 0.001 

            # Compute the distance between the robot's end effector and the handle
            dist_to_handle = np.linalg.norm(obs[:3] - obs[4:7])

            # Compute the difference between the drawer's current and goal positions
            goal_diff = np.linalg.norm(obs[4:7] - obs[-3:])

            # Check gripper openness. If the robot is close to the handle and the gripper is not closed, penalize
            gripper_penalty = 0.0
            if dist_to_handle < 0.1 and obs[3] > -1:
                gripper_penalty = 1.0

            # Compute the regularization term for the robot's action
            action_reg = np.linalg.norm(action)

            # Compute the final reward as a weighted sum of the different components
            reward = -w1*dist_to_handle - w2*goal_diff - w3*gripper_penalty - w4*action_reg

            '''
            """
            Compute the reward for the task of opening a drawer.

            :param action: np.ndarray, the action taken by the robot
            :param obs: observation which might include positions and states of robot and objects
            :return: float, the calculated reward
            """
            # Constants for reward calculation
            distance_weight = -1.0  # Penalize distance
            gripper_weight = -0.5   # Penalize incorrect gripper openness
            action_weight = -0.01   # Small penalty on large actions to encourage smoothness
        
            # 1. Distance from gripper to the handle of the drawer (assuming obj1 is the handle)
            distance_to_handle = np.linalg.norm(obs[:3] - obs[4:7])
        
            # 2. Gripper openness penalty (assuming a value of 1 is fully open and appropriate for grasping)
            gripper_penalty = np.square(obs[3] - 1)
        
            # 3. Regularization of the action (encourage smaller actions)
            action_penalty = np.sum(np.square(action))
        
            # Calculate total reward
            reward = (distance_weight * distance_to_handle +
                    gripper_weight * gripper_penalty +
                    action_weight * action_penalty) '''
            handle = obs[4:7]

            handle_error = np.linalg.norm(handle - self._target_pos)
            return reward, handle_error
        elif self.reward_func_version == 't2r3':
            gripper_pos = obs[:3]
            gripper_openness = obs[3]
            handle_pos = obs[4:7]  # Assuming obj1 is the handle of the drawer

            # Constants to tune the reward function
            distance_scale = 1.0
            goal_scale = 2.0
            openness_reward_scale = 0.5
            action_magnitude_scale = 0.1

            # 1. Distance to Handle Reward: Negative squared Euclidean distance between gripper and handle
            dist_to_handle = np.linalg.norm(gripper_pos - handle_pos)
            distance_reward = - (distance_scale * dist_to_handle ** 2)

            # 2. Goal State Reward: Negative squared distance between current drawer position and goal position
            drawer_pos = obs[11:14]  # Assuming obj2 is the drawer
            goal_distance = np.linalg.norm(drawer_pos - obs[-3:])
            goal_reward = - (goal_scale * goal_distance ** 2)

            # 3. Gripper Openness Reward: Encourage the gripper to be closed when near the handle (simple heuristic)
            if dist_to_handle < 0.1:  # Assuming 0.1 as a threshold distance to consider "near" the handle
                gripper_openness_reward = -abs(gripper_openness + 1) * openness_reward_scale  # Reward for having t>
            else:
                gripper_openness_reward = -abs(gripper_openness) * openness_reward_scale  # Reward for having the g>

            # 4. Action Magnitude Penalty: Discourage too large actions (energy efficiency)
            action_magnitude_penalty = -action_magnitude_scale * np.linalg.norm(action)

            # Total Reward
            total_reward = distance_reward + goal_reward + gripper_openness_reward + action_magnitude_penalty
            handle = obs[4:7]

            handle_error = np.linalg.norm(handle - self._target_pos)
            return total_reward, handle_error
