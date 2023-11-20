import mujoco
import numpy as np
from gymnasium.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerFaucetOpenEnvV2(SawyerXYZEnv):
    def __init__(self, render_mode=None, reward_func_version="v2"):
        hand_low = (-0.5, 0.40, -0.15)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.05, 0.8, 0.0)
        obj_high = (0.05, 0.85, 0.0)
        self._handle_length = 0.175
        self._target_radius = 0.07

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
        )

        self.reward_func_version = reward_func_version

        self.init_config = {
            "obj_init_pos": np.array([0, 0.8, 0.0]),
            "hand_init_pos": np.array([0.0, 0.4, 0.2]),
        }
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
        return full_v2_path_for("sawyer_xyz/sawyer_faucet.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (reward, target_to_obj) = self.compute_reward(action, obs)

        info = {"success": float(target_to_obj <= 0.07)}

        return reward, info

    @property
    def _target_site_config(self):
        return [
            ("goal_open", self._target_pos),
            ("goal_close", np.array([10.0, 10.0, 10.0])),
        ]

    def _get_pos_objects(self):
        return self._get_site_pos("handleStartOpen") + np.array([0.0, 0.0, -0.01])

    def _get_quat_objects(self):
        return self.data.body("faucetBase").xquat

    def reset_model(self):
        self._reset_hand()

        # Compute faucet position
        self.obj_init_pos = self._get_state_rand_vec()
        # Set mujoco body to computed position
        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "faucetBase")
        ] = self.obj_init_pos

        self._target_pos = self.obj_init_pos + np.array(
            [+self._handle_length, 0.0, 0.125]
        )
        mujoco.mj_forward(self.model, self.data)

        self.maxPullDist = np.linalg.norm(self._target_pos - self.obj_init_pos)

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()
        self.reachCompleted = False

    def compute_reward(self, action, obs):
        if self.reward_func_version == "v2":
            del action
            obj = obs[4:7] + np.array([-0.04, 0.0, 0.03])
            tcp = self.tcp_center
            target = self._target_pos.copy()

            target_to_obj = obj - target
            target_to_obj = np.linalg.norm(target_to_obj)
            target_to_obj_init = self.obj_init_pos - target
            target_to_obj_init = np.linalg.norm(target_to_obj_init)

            in_place = reward_utils.tolerance(
                target_to_obj,
                bounds=(0, self._target_radius),
                margin=abs(target_to_obj_init - self._target_radius),
                sigmoid="long_tail",
            )

            faucet_reach_radius = 0.01
            tcp_to_obj = np.linalg.norm(obj - tcp)
            tcp_to_obj_init = np.linalg.norm(self.obj_init_pos - self.init_tcp)
            reach = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, faucet_reach_radius),
                margin=abs(tcp_to_obj_init - faucet_reach_radius),
                sigmoid="gaussian",
            )

            tcp_opened = 0
            object_grasped = reach

            reward = 2 * reach + 3 * in_place

            reward *= 2

            reward = 10 if target_to_obj <= self._target_radius else reward

            return (reward, target_to_obj)
        else:
            del action

            objPos = obs[4:7]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            pullGoal = self._target_pos

            pullDist = np.linalg.norm(objPos - pullGoal)
            reachDist = np.linalg.norm(objPos - fingerCOM)
            reachRew = -reachDist

            self.reachCompleted = reachDist < 0.05

            def pullReward():
                c1 = 1000
                c2 = 0.01
                c3 = 0.001

                if self.reachCompleted:
                    pullRew = 1000 * (self.maxPullDist - pullDist) + c1 * (
                        np.exp(-(pullDist**2) / c2) + np.exp(-(pullDist**2) / c3)
                    )
                    pullRew = max(pullRew, 0)
                    return pullRew
                else:
                    return 0

            pullRew = pullReward()
            reward = reachRew + pullRew

            return [reward, pullDist]
