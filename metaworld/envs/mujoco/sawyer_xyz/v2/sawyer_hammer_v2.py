import mujoco
import numpy as np
from gymnasium.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerHammerEnvV2(SawyerXYZEnv):
    HAMMER_HANDLE_LENGTH = 0.14

    def __init__(self, render_mode=None, reward_func_version="v2"):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.4, 0.0)
        obj_high = (0.1, 0.5, 0.0)
        goal_low = (0.2399, 0.7399, 0.109)
        goal_high = (0.2401, 0.7401, 0.111)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
        )

        self.reward_func_version = reward_func_version

        self.init_config = {
            "hammer_init_pos": np.array([0, 0.5, 0.0]),
            "hand_init_pos": np.array([0, 0.4, 0.2]),
        }
        self.goal = self.init_config["hammer_init_pos"]
        self.hammer_init_pos = self.init_config["hammer_init_pos"]
        self.obj_init_pos = self.hammer_init_pos.copy()
        self.hand_init_pos = self.init_config["hand_init_pos"]
        self.nail_init_pos = None

        self._random_reset_space = Box(np.array(obj_low), np.array(obj_high))
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_hammer.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            success,
        ) = self.compute_reward(action, obs)

        info = {"success": float(success)}

        return reward, info

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id("HammerHandle")

    def _get_pos_objects(self):
        return np.hstack(
            (self.get_body_com("hammer").copy(), self.get_body_com("nail_link").copy())
        )

    def _get_quat_objects(self):
        return np.hstack(
            (self.data.body("hammer").xquat, self.data.body("nail_link").xquat)
        )

    def _set_hammer_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()

        # Set position of box & nail (these are not randomized)
        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        ] = np.array([0.24, 0.85, 0.0])
        # Update _target_pos
        self._target_pos = self._get_site_pos("goal")

        # Randomize hammer position
        self.hammer_init_pos = self._get_state_rand_vec()
        self.nail_init_pos = self._get_site_pos("nailHead")
        self.obj_init_pos = self.hammer_init_pos.copy()
        self._set_hammer_xyz(self.hammer_init_pos)

        self.liftThresh = 0.09
        self.hammerHeight = self.get_body_com("hammer").copy()[2]
        self.heightTarget = self.hammerHeight + self.liftThresh

        self.maxHammerDist = (
            np.linalg.norm(
                np.array(
                    [
                        self.hammer_init_pos[0],
                        self.hammer_init_pos[1],
                        self.heightTarget,
                    ]
                )
                - np.array(self.obj_init_pos)
            )
            + self.heightTarget
            + np.abs(self.obj_init_pos[1] - self._target_pos[1])
        )

        self.pickCompleted = False

        return self._get_obs()

    @staticmethod
    def _reward_quat(obs):
        # Ideal laid-down wrench has quat [1, 0, 0, 0]
        # Rather than deal with an angle between quaternions, just approximate:
        ideal = np.array([1.0, 0.0, 0.0, 0.0])
        error = np.linalg.norm(obs[7:11] - ideal)
        return max(1.0 - error / 0.4, 0.0)

    @staticmethod
    def _reward_pos(hammer_head, target_pos):
        pos_error = target_pos - hammer_head

        a = 0.1  # Relative importance of just *trying* to lift the hammer
        b = 0.9  # Relative importance of hitting the nail
        lifted = hammer_head[2] > 0.02
        in_place = a * float(lifted) + b * reward_utils.tolerance(
            np.linalg.norm(pos_error),
            bounds=(0, 0.02),
            margin=0.2,
            sigmoid="long_tail",
        )

        return in_place

    def compute_reward(self, actions, obs):
        if self.reward_func_version == "v2":
            hand = obs[:3]
            hammer = obs[4:7]
            hammer_head = hammer + np.array([0.16, 0.06, 0.0])
            # `self._gripper_caging_reward` assumes that the target object can be
            # approximated as a sphere. This is not true for the hammer handle, so
            # to avoid re-writing the `self._gripper_caging_reward` we pass in a
            # modified hammer position.
            # This modified position's X value will perfect match the hand's X value
            # as long as it's within a certain threshold
            hammer_threshed = hammer.copy()
            threshold = SawyerHammerEnvV2.HAMMER_HANDLE_LENGTH / 2.0
            if abs(hammer[0] - hand[0]) < threshold:
                hammer_threshed[0] = hand[0]

            reward_quat = SawyerHammerEnvV2._reward_quat(obs)
            reward_grab = self._gripper_caging_reward(
                actions,
                hammer_threshed,
                object_reach_radius=0.01,
                obj_radius=0.015,
                pad_success_thresh=0.02,
                xz_thresh=0.01,
                high_density=True,
            )
            reward_in_place = SawyerHammerEnvV2._reward_pos(
                hammer_head, self._target_pos
            )

            reward = (2.0 * reward_grab + 6.0 * reward_in_place) * reward_quat
            # Override reward on success. We check that reward is above a threshold
            # because this env's success metric could be hacked easily
            success = self.data.joint("NailSlideJoint").qpos > 0.09
            if success and reward > 5.0:
                reward = 10.0

            return (reward, success)
        else:
            hammerPos = obs[4:7]
            hammerHeadPos = self.data.geom("HammerHead").xpos.copy()
            objPos = self.data.site("nailHead").xpos

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            heightTarget = self.heightTarget

            hammerDist = np.linalg.norm(objPos - hammerHeadPos)
            screwDist = np.abs(objPos[1] - self._target_pos[1])
            reachDist = np.linalg.norm(hammerPos - fingerCOM)

            reachRew = -reachDist
            # incentive to close fingers when reachDist is small
            if reachDist < 0.05:
                reachRew = -reachDist + max(actions[-1], 0) / 50


            tolerance = 0.01
            if hammerPos[2] >= (heightTarget - tolerance):
                self.pickCompleted = True
            else:
                self.pickCompleted = False


            objDropped = (
                    (hammerPos[2] < (self.hammerHeight + 0.005))
                    and (hammerDist > 0.02)
                    and (reachDist > 0.02)
                )
                # Object on the ground, far away from the goal, and from the gripper
                # Can tweak the margin limits

            hScale = 100

            if self.pickCompleted and not (objDropped()):
                pickRew = hScale * heightTarget
            elif (reachDist < 0.1) and (hammerPos[2] > (self.hammerHeight + 0.005)):
                pickRew = hScale * min(heightTarget, hammerPos[2])
            else:
                pickRew = 0

            c1 = 1000
            c2 = 0.01
            c3 = 0.001

            cond = self.pickCompleted and (reachDist < 0.1) and not (objDropped())
            if cond:
                hammerRew = 1000 * (
                    self.maxHammerDist - hammerDist - screwDist
                ) + c1 * (
                    np.exp(-((hammerDist + screwDist) ** 2) / c2)
                    + np.exp(-((hammerDist + screwDist) ** 2) / c3)
                )
                hammerRew = max(hammerRew, 0)
            else:
                hammerRew, hammerDist, screwDist = [0, hammerDist, screwDist]

            assert (hammerRew >= 0) and (pickRew >= 0)
            reward = reachRew + pickRew + hammerRew
            success = self.data.joint("NailSlideJoint").qpos > 0.09
            return [reward, success]
