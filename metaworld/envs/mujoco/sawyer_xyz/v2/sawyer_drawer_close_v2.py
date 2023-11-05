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

    def __init__(self, render_mode=None, reward_func_version='v2'):
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
        obj = obs[4:7]
        if self.reward_func_version == 'v2':
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

            return (reward, tcp_to_obj, tcp_opened, target_to_obj, object_grasped, in_place)
        else:
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
                        np.exp(-(pullDist ** 2) / c2) + np.exp(-(pullDist ** 2) / c3)
                )
                pullRew = max(pullRew, 0)
            else:
                pullRew = 0

            reward = -reachDist + pullRew

            return [reward, pullDist]
