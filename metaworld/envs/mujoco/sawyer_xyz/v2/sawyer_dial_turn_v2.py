import mujoco
import numpy as np
from gymnasium.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerDialTurnEnvV2(SawyerXYZEnv):
    TARGET_RADIUS = 0.07

    def __init__(self, render_mode=None, reward_func_version='v2'):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.7, 0.0)
        obj_high = (0.1, 0.8, 0.0)
        goal_low = (-0.1, 0.73, 0.0299)
        goal_high = (0.1, 0.83, 0.0301)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
        )

        self.reward_func_version = reward_func_version

        self.init_config = {
            "obj_init_pos": np.array([0, 0.7, 0.0]),
            "hand_init_pos": np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0.0, 0.73, 0.08])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_dial.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            target_to_obj
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(target_to_obj <= self.TARGET_RADIUS)
        }

        return reward, info

    def _get_pos_objects(self):
        dial_center = self.get_body_com("dial").copy()
        dial_angle_rad = self.data.joint("knob_Joint_1").qpos

        offset = np.array(
            [np.sin(dial_angle_rad)[0], -np.cos(dial_angle_rad)[0], 0],
        )
        dial_radius = 0.05

        offset *= dial_radius

        return dial_center + offset

    def _get_quat_objects(self):
        return self.data.body("dial").xquat

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.prev_obs = self._get_curr_obs_combined_no_goal()

        goal_pos = self._get_state_rand_vec()
        self.obj_init_pos = goal_pos[:3]
        final_pos = goal_pos.copy() + np.array([0, 0.03, 0.03])
        self._target_pos = final_pos
        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "dial")
        ] = self.obj_init_pos
        self.dial_push_position = self._get_pos_objects() + np.array([0.05, 0.02, 0.09])
        mujoco.mj_forward(self.model, self.data)

        self.maxPullDist = np.abs(self._target_pos[1] - self.obj_init_pos[1])

        return self._get_obs()

    def compute_reward(self, action, obs):
        if self.reward_func_version == 'v2':
            obj = self._get_pos_objects()
            dial_push_position = self._get_pos_objects() + np.array([0.05, 0.02, 0.09])
            tcp = self.tcp_center
            target = self._target_pos.copy()

            target_to_obj = obj - target
            target_to_obj = np.linalg.norm(target_to_obj)
            target_to_obj_init = self.dial_push_position - target
            target_to_obj_init = np.linalg.norm(target_to_obj_init)

            in_place = reward_utils.tolerance(
                target_to_obj,
                bounds=(0, self.TARGET_RADIUS),
                margin=abs(target_to_obj_init - self.TARGET_RADIUS),
                sigmoid="long_tail",
            )

            dial_reach_radius = 0.005
            tcp_to_obj = np.linalg.norm(dial_push_position - tcp)
            tcp_to_obj_init = np.linalg.norm(self.dial_push_position - self.init_tcp)
            reach = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, dial_reach_radius),
                margin=abs(tcp_to_obj_init - dial_reach_radius),
                sigmoid="gaussian",
            )
            gripper_closed = min(max(0, action[-1]), 1)

            reach = reward_utils.hamacher_product(reach, gripper_closed)
            tcp_opened = 0
            object_grasped = reach

            reward = 10 * reward_utils.hamacher_product(reach, in_place)
            return (
                reward,
                target_to_obj
            )
        else:
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

            def pullReward():
                c1 = 1000
                c2 = 0.001
                c3 = 0.0001

                if self.reachCompleted:
                    pullRew = 1000 * (self.maxPullDist - pullDist) + c1 * (
                            np.exp(-(pullDist ** 2) / c2) + np.exp(-(pullDist ** 2) / c3)
                    )
                    pullRew = max(pullRew, 0)
                    return pullRew
                else:
                    return 0

            pullRew = pullReward()
            reward = reachRew + pullRew

            return [reward, pullDist]


 


 
