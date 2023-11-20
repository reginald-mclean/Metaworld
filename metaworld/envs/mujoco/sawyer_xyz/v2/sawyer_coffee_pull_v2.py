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


class SawyerCoffeePullEnvV2(SawyerXYZEnv):
    def __init__(self, render_mode=None, reward_func_version='v2'):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.05, 0.7, -0.001)
        obj_high = (0.05, 0.75, +0.001)
        goal_low = (-0.1, 0.55, -0.001)
        goal_high = (0.1, 0.65, +0.001)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
        )

        self.reward_func_version = reward_func_version

        self.init_config = {
            "obj_init_pos": np.array([0, 0.75, 0.0]),
            "obj_init_angle": 0.3,
            "hand_init_pos": np.array([0.0, 0.4, 0.2]),
        }
        self.goal = np.array([0.0, 0.6, 0])
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
        return full_v2_path_for("sawyer_xyz/sawyer_coffee.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            obj_to_target
        ) = self.compute_reward(action, obs)
        success = float(obj_to_target <= 0.07)

        info = {
            "success": success
        }

        return reward, info

    @property
    def _target_site_config(self):
        return [("mug_goal", self._target_pos)]

    def _get_pos_objects(self):
        return self.get_body_com("obj")

    def _get_quat_objects(self):
        geom_xmat = self.data.geom("mug").xmat.reshape(3, 3)
        return Rotation.from_matrix(geom_xmat).as_quat()

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()
        qpos[0:3] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()

        pos_mug_init, pos_mug_goal = np.split(self._get_state_rand_vec(), 2)
        while np.linalg.norm(pos_mug_init[:2] - pos_mug_goal[:2]) < 0.15:
            pos_mug_init, pos_mug_goal = np.split(self._get_state_rand_vec(), 2)

        self._set_obj_xyz(pos_mug_init)
        self.obj_init_pos = pos_mug_init

        pos_machine = pos_mug_init + np.array([0.0, 0.22, 0.0])
        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "coffee_machine")
        ] = pos_machine

        self._target_pos = pos_mug_goal

        self.maxPullDist = np.linalg.norm(
            self.obj_init_pos[:2] - np.array(self._target_pos)[:2]
        )

        return self._get_obs()

    def compute_reward(self, action, obs):
        if self.reward_func_version == 'v2':
            obj = obs[4:7]
            target = self._target_pos.copy()

            # Emphasize X and Y errors
            scale = np.array([2.0, 2.0, 1.0])
            target_to_obj = (obj - target) * scale
            target_to_obj = np.linalg.norm(target_to_obj)
            target_to_obj_init = (self.obj_init_pos - target) * scale
            target_to_obj_init = np.linalg.norm(target_to_obj_init)

            in_place = reward_utils.tolerance(
                target_to_obj,
                bounds=(0, 0.05),
                margin=target_to_obj_init,
                sigmoid="long_tail",
            )
            tcp_opened = obs[3]
            tcp_to_obj = np.linalg.norm(obj - self.tcp_center)

            object_grasped = self._gripper_caging_reward(
                action,
                obj,
                object_reach_radius=0.04,
                obj_radius=0.02,
                pad_success_thresh=0.05,
                xz_thresh=0.05,
                desired_gripper_effort=0.7,
                medium_density=True,
            )

            reward = reward_utils.hamacher_product(object_grasped, in_place)

            if tcp_to_obj < 0.04 and tcp_opened > 0:
                reward += 1.0 + 5.0 * in_place
            if target_to_obj < 0.05:
                reward = 10.0
            return (
                reward,
                np.linalg.norm(obj - target),  # recompute to avoid `scale` above
            )
        else:
            objPos = obs[4:7]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            goal = self._target_pos

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            assert np.all(goal == self._get_site_pos("mug_goal"))
            reachDist = np.linalg.norm(fingerCOM - objPos)
            pullDist = np.linalg.norm(objPos[:2] - goal[:2])
            reachRew = -reachDist
            reachDistxy = np.linalg.norm(
                np.concatenate((objPos[:-1], [self.init_tcp[-1]])) - fingerCOM
            )

            if reachDistxy < 0.05:  # 0.02
                reachRew = -reachDist + 0.1
                if reachDist < 0.05:
                    reachRew += max(action[-1], 0) / 50
            else:
                reachRew = -reachDistxy

            if reachDist < 0.05:
                pullRew = 1000 * (self.maxPullDist - pullDist) + c1 * (
                        np.exp(-(pullDist ** 2) / c2) + np.exp(-(pullDist ** 2) / c3)
                )
                pullRew = max(pullRew, 0)
            else:
                pullRew = 0

            reward = reachRew + pullRew

            return [reward, pullDist]


 


 
