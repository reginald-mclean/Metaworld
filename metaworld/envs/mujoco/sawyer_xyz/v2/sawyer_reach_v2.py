import mujoco
import numpy as np
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerReachEnvV2(SawyerXYZEnv):
    """SawyerReachEnv.

    Motivation for V2:
        V1 was very difficult to solve because the observation didn't say where
        to move (where to reach).
    Changelog from V1 to V2:
        - (7/7/20) Removed 3 element vector. Replaced with 3 element position
            of the goal (for consistency with other environments)
        - (6/15/20) Added a 3 element vector to the observation. This vector
            points from the end effector to the goal coordinate.
            i.e. (self._target_pos - pos_hand)
        - (6/15/20) Separated reach-push-pick-place into 3 separate envs.
    """

    def __init__(self, tasks=None, render_mode=None, reward_func_version="v2", camera_name=None, camera_id=None):
        goal_low = (-0.1, 0.8, 0.05)
        goal_high = (0.1, 0.9, 0.3)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.02)
        obj_high = (0.1, 0.7, 0.02)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id
        )

        self.reward_func_version = reward_func_version

        self.init_config = {
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0.0, 0.6, 0.02]),
            "hand_init_pos": np.array([0.0, 0.6, 0.2]),
        }

        self.goal = np.array([-0.1, 0.8, 0.2])

        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_reach_v2.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        reward, reach_dist = self.compute_reward(action, obs)
        success = float(reach_dist <= 0.05)

        info = {"success": success}

        return reward, info

    def _get_pos_objects(self):
        return self.get_body_com("obj")

    def _get_quat_objects(self):
        geom_xmat = self.data.geom("objGeom").xmat.reshape(3, 3)
        return Rotation.from_matrix(geom_xmat).as_quat()

    def fix_extreme_obj_pos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not
        # aligned. If this is not done, the object could be initialized in an
        # extreme position
        diff = self.get_body_com("obj")[:2] - self.get_body_com("obj")[:2]
        adjusted_pos = orig_init_pos[:2] + diff
        # The convention we follow is that body_com[2] is always 0,
        # and geom_pos[2] is the object height
        return [adjusted_pos[0], adjusted_pos[1], self.get_body_com("obj")[-1]]

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.fix_extreme_obj_pos(self.init_config["obj_init_pos"])
        self.obj_init_angle = self.init_config["obj_init_angle"]

        goal_pos = self._get_state_rand_vec()
        self._target_pos = goal_pos[3:]
        while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = goal_pos[3:]
        self._target_pos = goal_pos[-3:]
        self.obj_init_pos = goal_pos[:3]
        self._set_obj_xyz(self.obj_init_pos)
        mujoco.mj_forward(self.model, self.data)
        self.maxReachDist = np.linalg.norm(self.init_tcp - np.array(self._target_pos))
        return self._get_obs()

    def compute_reward(self, actions, obs):
        if self.reward_func_version == "v2":
            _TARGET_RADIUS = 0.05
            tcp = self.tcp_center
            # obj = obs[4:7]
            # tcp_opened = obs[3]
            target = self._target_pos

            tcp_to_target = np.linalg.norm(tcp - target)
            # obj_to_target = np.linalg.norm(obj - target)

            in_place_margin = np.linalg.norm(self.hand_init_pos - target)
            in_place = reward_utils.tolerance(
                tcp_to_target,
                bounds=(0, _TARGET_RADIUS),
                margin=in_place_margin,
                sigmoid="long_tail",
            )

            return [10 * in_place, tcp_to_target]
        elif self.reward_func_version == 'v1':
            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2
            goal = self._target_pos

            del actions
            del obs

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            reachDist = np.linalg.norm(fingerCOM - goal)
            reachRew = c1 * (self.maxReachDist - reachDist) + c1 * (
                np.exp(-(reachDist**2) / c2) + np.exp(-(reachDist**2) / c3)
            )
            reachRew = max(reachRew, 0)
            reward = reachRew
            return [reward, reachDist]
        elif self.reward_func_version == 'text2reward':
            # Calculate the distance between the end-effector and the goal position
            distance_to_goal = np.linalg.norm(obs[:3] - obs[-3:])

            # Regularize the action to discourage large movements
            action_cost = np.sum(np.square(actions))

            # Formulate the reward as a weighted sum of the negative distance (to maximize closeness)
            # and the negative square sum of the action (to encourage smaller actions)
            reward = -distance_to_goal - 0.1 * action_cost

            return reward, distance_to_goal
        elif self.reward_func_version == 't2r3':
            # Extract the current position of the robot's end-effector
            current_ee_position = obs[:3]

            # Calculate the Euclidean distance to the goal position
            distance_to_goal = np.linalg.norm(current_ee_position - obs[-3:])

            reward = -distance_to_goal**2
    
            return reward, distance_to_goal

