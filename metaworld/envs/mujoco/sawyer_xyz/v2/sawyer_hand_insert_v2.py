import numpy as np
from gymnasium.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerHandInsertEnvV2(SawyerXYZEnv):
    TARGET_RADIUS = 0.05

    def __init__(self, render_mode=None, reward_func_version='v2'):
        hand_low = (-0.5, 0.40, -0.15)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.05)
        obj_high = (0.1, 0.7, 0.05)
        goal_low = (-0.04, 0.8, -0.0201)
        goal_high = (0.04, 0.88, -0.0199)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
        )

        self.reward_func_version = reward_func_version

        self.init_config = {
            "obj_init_pos": np.array([0, 0.6, 0.05]),
            "obj_init_angle": 0.3,
            "hand_init_pos": np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0.0, 0.84, -0.08], dtype=np.float32)
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_table_with_hole.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        obj = obs[4:7]

        (
            reward,
            obj_to_target
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(obj_to_target <= 0.05)
        }

        return reward, info

    @property
    def _get_id_main_object(self):
        return self.model.geom("objGeom").id

    def _get_pos_objects(self):
        return self.get_body_com("obj")

    def _get_quat_objects(self):
        return self.data.body("obj").xquat

    def reset_model(self):
        self._reset_hand()
        self.prev_obs = self._get_curr_obs_combined_no_goal()
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.objHeight = self.get_body_com("obj")[2]

        goal_pos = self._get_state_rand_vec()
        while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.15:
            goal_pos = self._get_state_rand_vec()
        self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))
        self._target_pos = goal_pos[-3:]

        self._set_obj_xyz(self.obj_init_pos)

        self.maxReachDist = np.abs(self.hand_init_pos[-1] - self._target_pos[-1])

        return self._get_obs()

    def compute_reward(self, action, obs):
        if self.reward_func_version == 'v2':
            obj = obs[4:7]

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
            reward = reward_utils.hamacher_product(object_grasped, in_place)

            tcp_opened = obs[3]
            tcp_to_obj = np.linalg.norm(obj - self.tcp_center)

            if tcp_to_obj < 0.02 and tcp_opened > 0:
                reward += 1.0 + 7.0 * in_place
            if target_to_obj < self.TARGET_RADIUS:
                reward = 10.0
            return (reward, target_to_obj)
        else:
            del action

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            goal = self._target_pos

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            reachDist = np.linalg.norm(fingerCOM[:-1] - goal[:-1])
            reachRew = -reachDist
            reachDist_z = np.abs(fingerCOM[-1] - goal[-1])

            if reachDist < 0.05:
                reachNearRew = 1000 * (self.maxReachDist - reachDist_z) + c1 * (
                        np.exp(-(reachDist_z ** 2) / c2) + np.exp(-(reachDist_z ** 2) / c3)
                )
            else:
                reachNearRew = 0.0

            reachNearRew = max(reachNearRew, 0)
            reward = reachRew + reachNearRew

            return [reward, reachDist]


 


 
