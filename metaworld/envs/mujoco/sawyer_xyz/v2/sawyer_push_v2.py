import numpy as np
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerPushEnvV2(SawyerXYZEnv):
    """SawyerPushEnv.

    Motivation for V2:
        V1 was very difficult to solve because the observation didn't say where
        to move after reaching the puck.
    Changelog from V1 to V2:
        - (7/7/20) Removed 3 element vector. Replaced with 3 element position
            of the goal (for consistency with other environments)
        - (6/15/20) Added a 3 element vector to the observation. This vector
            points from the end effector to the goal coordinate.
            i.e. (self._target_pos - pos_hand)
        - (6/15/20) Separated reach-push-pick-place into 3 separate envs.
    """

    TARGET_RADIUS = 0.05

    def __init__(self, tasks=None, render_mode=None, reward_func_version="v2"):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.02)
        obj_high = (0.1, 0.7, 0.02)
        goal_low = (-0.1, 0.8, 0.01)
        goal_high = (0.1, 0.9, 0.02)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
        )

        self.reward_func_version = reward_func_version

        self.init_config = {
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0.0, 0.6, 0.02]),
            "hand_init_pos": np.array([0.0, 0.6, 0.2]),
        }

        self.goal = np.array([0.1, 0.8, 0.02])

        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([+1, +1, +1, +1]),
        )

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        self.num_resets = 0

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_push_v2.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (reward, target_to_obj) = self.compute_reward(action, obs)

        info = {
            "success": float(target_to_obj <= self.TARGET_RADIUS),
        }

        return reward, info

    def _get_quat_objects(self):
        geom_xmat = self.data.geom("objGeom").xmat.reshape(3, 3)
        return Rotation.from_matrix(geom_xmat).as_quat()

    def _get_pos_objects(self):
        return self.get_body_com("obj")

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
        self.pickCompleted = False
        self._target_pos = self.goal.copy()
        self.obj_init_pos = np.array(
            self.fix_extreme_obj_pos(self.init_config["obj_init_pos"])
        )
        self.obj_init_angle = self.init_config["obj_init_angle"]

        goal_pos = self._get_state_rand_vec()
        self._target_pos = goal_pos[3:]
        while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = goal_pos[3:]
        self._target_pos = np.concatenate((goal_pos[-3:-1], [self.obj_init_pos[-1]]))
        self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))

        self._set_obj_xyz(self.obj_init_pos)
        self.objHeight = self.data.geom("objGeom").xpos[2]
        self.heightTarget = self.objHeight + 0.04
        self.maxPushDist = np.linalg.norm(
            self.obj_init_pos[:2] - np.array(self._target_pos)[:2]
        )
        self.maxPlacingDist = (
            np.linalg.norm(
                np.array(
                    [self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]
                )
                - np.array(self._target_pos)
            )
            + self.heightTarget
        )
        return self._get_obs()

    def compute_reward(self, action, obs):
        if self.reward_func_version == "v2":
            obj = obs[4:7]
            tcp_opened = obs[3]
            tcp_to_obj = np.linalg.norm(obj - self.tcp_center)
            target_to_obj = np.linalg.norm(obj - self._target_pos)
            target_to_obj_init = np.linalg.norm(self.obj_init_pos - self._target_pos)

            in_place = reward_utils.tolerance(
                target_to_obj,
                bounds=(0, self.TARGET_RADIUS),
                margin=target_to_obj_init,
                sigmoid="long_tail",
            )

            object_grasped = self._gripper_caging_reward(
                action,
                obj,
                object_reach_radius=0.01,
                obj_radius=0.015,
                pad_success_thresh=0.05,
                xz_thresh=0.005,
                high_density=True,
            )
            reward = 2 * object_grasped

            if tcp_to_obj < 0.02 and tcp_opened > 0:
                reward += 1.0 + reward + 5.0 * in_place
            if target_to_obj < self.TARGET_RADIUS:
                reward = 10.0
            return (reward, target_to_obj)
        elif self.reward_func_version == 'v1':
            objPos = obs[4:7]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            heightTarget = self.heightTarget
            goal = self._target_pos

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            del action
            del obs

            assert np.all(goal == self._get_site_pos("goal"))
            reachDist = np.linalg.norm(fingerCOM - objPos)
            pushDist = np.linalg.norm(objPos[:2] - goal[:2])
            reachRew = -reachDist
            if reachDist < 0.05:
                pushRew = 1000 * (self.maxPushDist - pushDist) + c1 * (
                    np.exp(-(pushDist**2) / c2) + np.exp(-(pushDist**2) / c3)
                )
                pushRew = max(pushRew, 0)
            else:
                pushRew = 0
            reward = reachRew + pushRew
            return [reward, pushDist]
        elif self.reward_func_version == 'text2reward':
            # Extract relevant information from the environment state
            obj_position = obs[4:7]  # Position of the object to be pushed
            goal_position = obs[-3:] # The desired goal position
            action_magnitude = np.linalg.norm(action) # Magnitude of the action vector

            # 1. Distance reward: negative squared Euclidean distance between object and goal
            distance_to_goal = np.linalg.norm(obj_position - goal_position)
            distance_reward = -np.square(distance_to_goal)

            # 2. Action regularization: Penalize large actions to encourage smooth movements
            action_penalty = -0.01 * np.square(action_magnitude)  # Small weight to action penalty

            # Total reward is a combination of distance reward and action penalty
            total_reward = distance_reward + action_penalty

            return total_reward, distance_to_goal


