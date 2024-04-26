import mujoco
import numpy as np
from gymnasium.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerButtonPressTopdownEnvV2(SawyerXYZEnv):
    def __init__(self, tasks=None, render_mode=None, reward_func_version="v2"):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.8, 0.115)
        obj_high = (0.1, 0.9, 0.115)

        self.reward_func_version = reward_func_version

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
        )

        self.init_config = {
            "obj_init_pos": np.array([0, 0.8, 0.115], dtype=np.float32),
            "hand_init_pos": np.array([0, 0.4, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0, 0.88, 0.1])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_button_press_topdown.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (reward, obj_to_target) = self.compute_reward(action, obs)
        info = {
            "success": float(obj_to_target <= 0.024),
        }

        return reward, info

    @property
    def _target_site_config(self):
        return []

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id("btnGeom")

    def _get_pos_objects(self):
        return self.get_body_com("button") + np.array([0.0, 0.0, 0.193])

    def _get_quat_objects(self):
        return self.data.body("button").xquat

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        goal_pos = self._get_state_rand_vec()
        self.obj_init_pos = goal_pos
        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        ] = self.obj_init_pos
        mujoco.mj_forward(self.model, self.data)
        self._target_pos = self._get_site_pos("hole")

        self._obj_to_target_init = abs(
            self._target_pos[2] - self._get_site_pos("buttonStart")[2]
        )

        self.maxDist = np.abs(
            self._get_site_pos("buttonStart")[2] - self._target_pos[2]
        )

        return self._get_obs()

    def compute_reward(self, action, obs):
        if self.reward_func_version == "v2":
            obj = obs[4:7]
            tcp = self.tcp_center

            tcp_to_obj = np.linalg.norm(obj - tcp)
            tcp_to_obj_init = np.linalg.norm(obj - self.init_tcp)
            obj_to_target = abs(self._target_pos[2] - obj[2])

            tcp_closed = 1 - obs[3]
            near_button = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, 0.01),
                margin=tcp_to_obj_init,
                sigmoid="long_tail",
            )
            button_pressed = reward_utils.tolerance(
                obj_to_target,
                bounds=(0, 0.005),
                margin=self._obj_to_target_init,
                sigmoid="long_tail",
            )
            reward = 5 * reward_utils.hamacher_product(tcp_closed, near_button)
            if tcp_to_obj <= 0.03:
                reward += 5 * button_pressed

            return (reward, obj_to_target)
        elif self.reward_func_version == 'v1':
            objPos = obs[4:7]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            pressGoal = self._target_pos[2]

            pressDist = np.abs(objPos[2] - pressGoal)
            reachDist = np.linalg.norm(objPos - fingerCOM)
            reachRew = -reachDist

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            if reachDist < 0.05:
                pressRew = 1000 * (self.maxDist - pressDist) + c1 * (
                    np.exp(-(pressDist**2) / c2) + np.exp(-(pressDist**2) / c3)
                )
            else:
                pressRew = 0
            pressRew = max(pressRew, 0)
            reward = reachRew + pressRew

            return [reward, pressDist]
        elif self.reward_func_version == 'text2reward':
            ee_pos = obs[:3]
            goal_pos = obs[-3:]

            # 1. Distance metric (focusing on z-coordinate for pressing action)
            # Using the squared difference for smoother gradients and penalization
            z_distance = (ee_pos[2] - goal_pos[2]) ** 2

            # 2. Action regularization
            # Penalize larger actions to encourage minimal and smooth movements
            action_cost = np.sum(np.square(action))

            # Construct the reward function
            # Note: We use negative rewards for cost terms (distance and action)
            reward = -z_distance - 0.01 * action_cost  # scale action cost to reduce its dominance
            obj = obs[4:7]
            return reward, np.linalg.norm(self._target_pos[2] - obj[2])
        elif self.reward_func_version == 'text2reward2':
            ee_pos = obs[:3]  # End-effector position
            goal_pos = obs[-3:]    # Goal position (button location)

            # Part 1: Horizontal Distance Reward
            # Calculate the horizontal (x, y) distance and penalize it to encourage alignment over the button
            horizontal_distance = np.linalg.norm(ee_pos[:2] - goal_pos[:2])
            horizontal_distance_reward = -horizontal_distance

            # Part 2: Vertical Alignment Reward
            # Specifically focus on the z-axis alignment, rewarding the robot for being exactly above the button before descending
            vertical_distance = np.abs(ee_pos[2] - goal_pos[2])  # Absolute vertical distance to the goal
            vertical_alignment_reward = -vertical_distance

            # Part 3: Regularization on the action
            # Encourage smaller and smoother actions by penalizing the square of action values
            action_penalty = -np.sum(np.square(action))

            # Total reward
            # We use different weights for horizontal alignment and vertical positioning to prioritize vertical alignment as per task requirements
            total_reward = 0.5 * horizontal_distance_reward + vertical_alignment_reward + 0.01 * action_penalty

            obj = obs[4:7]
            return total_reward, np.linalg.norm(self._target_pos[2] - obj[2])
        elif self.reward_func_version == 't2r3':
            # Extract necessary components from the observation
            ee_position = obs[:3]  # End-effector position
            goal_position = obs[-3:]  # Target position for pressing the button
    
            # Calculate the Euclidean distance between the end-effector and the goal position
            distance_to_goal = np.linalg.norm(ee_position - goal_position)
    
            # Reward for getting closer to the goal
            # We use negative distance as we want the reward to increase as the distance decreases
            # Additionally, we can scale this or use an exponential to shape the reward function
            reward_distance = -distance_to_goal
    
            # If the distance is very small, we can assume the button can be pressed
            # We give a large positive reward if the end-effector is very close to the goal position
            if distance_to_goal < 0.05:  # Threshold for being close enough to press the button
                reward_press = 100  # Large reward for achieving the main task goal
            else:
                reward_press = 0
    
            # Total reward is a combination of the distance reward and the button press reward
            total_reward = reward_distance + reward_press
            obj = obs[4:7]
            return total_reward, np.linalg.norm(self._target_pos[2] - obj[2])


