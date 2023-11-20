import mujoco
import numpy as np
from gymnasium.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerNutDisassembleEnvV2(SawyerXYZEnv):
    WRENCH_HANDLE_LENGTH = 0.02

    def __init__(self, render_mode=None, reward_func_version='v2'):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0.0, 0.6, 0.025)
        obj_high = (0.1, 0.75, 0.02501)
        goal_low = (-0.1, 0.6, 0.1699)
        goal_high = (0.1, 0.75, 0.1701)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
        )

        self.reward_func_version = reward_func_version

        self.init_config = {
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0, 0.7, 0.025]),
            "hand_init_pos": np.array((0, 0.4, 0.2), dtype=np.float32),
        }
        self.goal = np.array([0, 0.8, 0.17])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(
            np.array(goal_low) + np.array([0.0, 0.0, 0.005]),
            np.array(goal_high) + np.array([0.0, 0.0, 0.005]),
        )

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_assembly_peg.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            success,
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(success)
        }

        return reward, info

    @property
    def _target_site_config(self):
        return [("pegTop", self._target_pos)]

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id("WrenchHandle")

    def _get_pos_objects(self):
        return self._get_site_pos("RoundNut-8")

    def _get_quat_objects(self):
        return self.data.body("RoundNut").xquat

    def _get_obs_dict(self):
        obs_dict = super()._get_obs_dict()
        obs_dict["state_achieved_goal"] = self.get_body_com("RoundNut")
        return obs_dict

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = np.array(self.init_config["obj_init_pos"])
        self.obj_init_angle = self.init_config["obj_init_angle"]

        goal_pos = self._get_state_rand_vec()
        while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
            goal_pos = self._get_state_rand_vec()
        self.obj_init_pos = goal_pos[:3]
        self._target_pos = goal_pos[:3] + np.array([0, 0, 0.15])

        peg_pos = self.obj_init_pos + np.array([0.0, 0.0, 0.03])
        peg_top_pos = self.obj_init_pos + np.array([0.0, 0.0, 0.08])
        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "peg")
        ] = peg_pos
        self.model.site_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pegTop")
        ] = peg_top_pos
        mujoco.mj_forward(self.model, self.data)
        self._set_obj_xyz(self.obj_init_pos)

        self.liftThresh = 0.05
        self.objHeight = self.data.body("RoundNut").xpos[2]
        self.heightTarget = self.objHeight + self.liftThresh
        self.maxPlacingDist = (
                np.linalg.norm(
                    np.array(
                        [self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]
                    )
                    - np.array(self._target_pos)
                )
                + self.heightTarget
        )
        self.pickCompleted = False
        return self._get_obs()

    @staticmethod
    def _reward_quat(obs):
        # Ideal laid-down wrench has quat [.707, 0, 0, .707]
        # Rather than deal with an angle between quaternions, just approximate:
        ideal = np.array([0.707, 0, 0, 0.707])
        error = np.linalg.norm(obs[7:11] - ideal)
        return max(1.0 - error / 0.4, 0.0)

    @staticmethod
    def _reward_pos(wrench_center, target_pos):
        pos_error = target_pos + np.array([0.0, 0.0, 0.1]) - wrench_center

        a = 0.1  # Relative importance of just *trying* to lift the wrench
        b = 0.9  # Relative importance of placing the wrench on the peg
        lifted = wrench_center[2] > 0.02
        in_place = a * float(lifted) + b * reward_utils.tolerance(
            np.linalg.norm(pos_error),
            bounds=(0, 0.02),
            margin=0.2,
            sigmoid="long_tail",
        )

        return in_place

    def compute_reward(self, actions, obs):
        if self.reward_func_version == 'v2':
            hand = obs[:3]
            wrench = obs[4:7]
            wrench_center = self._get_site_pos("RoundNut")
            # `self._gripper_caging_reward` assumes that the target object can be
            # approximated as a sphere. This is not true for the wrench handle, so
            # to avoid re-writing the `self._gripper_caging_reward` we pass in a
            # modified wrench position.
            # This modified position's X value will perfect match the hand's X value
            # as long as it's within a certain threshold
            wrench_threshed = wrench.copy()
            threshold = SawyerNutDisassembleEnvV2.WRENCH_HANDLE_LENGTH / 2.0
            if abs(wrench[0] - hand[0]) < threshold:
                wrench_threshed[0] = hand[0]

            reward_quat = SawyerNutDisassembleEnvV2._reward_quat(obs)
            reward_grab = self._gripper_caging_reward(
                actions,
                wrench_threshed,
                object_reach_radius=0.01,
                obj_radius=0.015,
                pad_success_thresh=0.02,
                xz_thresh=0.01,
                high_density=True,
            )
            reward_in_place = SawyerNutDisassembleEnvV2._reward_pos(
                wrench_center, self._target_pos
            )

            reward = (2.0 * reward_grab + 6.0 * reward_in_place) * reward_quat
            # Override reward on success
            success = obs[6] > self._target_pos[2]
            if success:
                reward = 10.0

            return (
                reward,
                success,
            )
        else:
            graspPos = obs[4:7]
            objPos = graspPos

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            heightTarget = self.heightTarget
            placingGoal = self._target_pos

            reachDist = np.linalg.norm(graspPos - fingerCOM)
            reachDistxy = np.linalg.norm(graspPos[:-1] - fingerCOM[:-1])
            zDist = np.abs(fingerCOM[-1] - self.init_tcp[-1])

            placingDist = np.linalg.norm(objPos - placingGoal)

            def reachReward():
                reachRew = -reachDist
                if reachDistxy < 0.04:
                    reachRew = -reachDist
                else:
                    reachRew = -reachDistxy - 2 * zDist

                # incentive to close fingers when reachDist is small
                if reachDist < 0.04:
                    reachRew = -reachDist + max(actions[-1], 0) / 50
                return reachRew, reachDist

            def pickCompletionCriteria():
                tolerance = 0.01
                if objPos[2] >= (heightTarget - tolerance) and reachDist < 0.04:
                    return True
                else:
                    return False

            if pickCompletionCriteria():
                self.pickCompleted = True

            def objDropped():
                return (
                        (objPos[2] < (self.objHeight + 0.005))
                        and (placingDist > 0.02)
                        and (reachDist > 0.02)
                )

            def orig_pickReward():
                hScale = 100
                if self.pickCompleted and not (objDropped()):
                    return hScale * heightTarget
                elif (reachDist < 0.04) and (objPos[2] > (self.objHeight + 0.005)):
                    return hScale * min(heightTarget, objPos[2])
                else:
                    return 0

            def placeRewardMove():
                c1 = 1000
                c2 = 0.01
                c3 = 0.001

                placeRew = 1000 * (self.maxPlacingDist - placingDist) + c1 * (
                        np.exp(-(placingDist ** 2) / c2) + np.exp(-(placingDist ** 2) / c3)
                )
                placeRew = max(placeRew, 0)
                cond = self.pickCompleted and (reachDist < 0.03) and not (objDropped())
                if cond:
                    return [placeRew, placingDist]
                else:
                    return [0, placingDist]

            reachRew, reachDist = reachReward()
            pickRew = orig_pickReward()

            peg_pos = self.model.body("peg").pos
            nut_pos = self.get_body_com("RoundNut")
            if abs(nut_pos[0] - peg_pos[0]) > 0.05 or abs(nut_pos[1] - peg_pos[1]) > 0.05:
                placingDist = 0
                reachRew = 0
                reachDist = 0
                pickRew = heightTarget * 100

            placeRew, placingDist = placeRewardMove()
            assert (placeRew >= 0) and (pickRew >= 0)
            reward = reachRew + pickRew + placeRew
            '''success = (
                              abs(nut_pos[0] - peg_pos[0]) > 0.05 or abs(nut_pos[1] - peg_pos[1]) > 0.05
                      ) or placingDist < 0.02'''
            success = obs[6] > self._target_pos[2]
            return [
                reward,
                float(success)
            ]


 


 
