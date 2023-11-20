import numpy as np
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerReachWallEnvV2(SawyerXYZEnv):
    """SawyerReachWallEnv.

    Motivation for V2:
        V1 was difficult to solve since the observations didn't say where
        to move (where to reach).
    Changelog from V1 to V2:
        - (7/7/20) Removed 3 element vector. Replaced with 3 element position
            of the goal (for consistency with other environments)
        - (6/17/20) Separated reach from reach-push-pick-place.
        - (6/17/20) Added a 3 element vector to the observation. This vector
            points from the end effector to the goal coordinate.
            i.e. (self._target_pos - pos_hand)
    """

    def __init__(self, render_mode=None, reward_func_version="v2"):
        goal_low = (-0.05, 0.85, 0.05)
        goal_high = (0.05, 0.9, 0.3)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.05, 0.6, 0.015)
        obj_high = (0.05, 0.65, 0.015)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
        )

        self.reward_func_version = reward_func_version

        self.init_config = {
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0, 0.6, 0.02]),
            "hand_init_pos": np.array([0, 0.6, 0.2]),
        }

        self.goal = np.array([-0.05, 0.8, 0.2])

        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.num_resets = 0

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_reach_wall_v2.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        reward, tcp_to_object = self.compute_reward(action, obs)
        success = float(tcp_to_object <= 0.05)

        info = {"success": success}

        return reward, info

    def _get_pos_objects(self):
        return self.get_body_com("obj")

    def _get_quat_objects(self):
        geom_xmat = self.data.geom("objGeom").xmat.reshape(3, 3)
        return Rotation.from_matrix(geom_xmat).as_quat()

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_angle = self.init_config["obj_init_angle"]

        goal_pos = self._get_state_rand_vec()
        self._target_pos = goal_pos[3:]
        while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = goal_pos[3:]
        self._target_pos = goal_pos[-3:]
        self.obj_init_pos = goal_pos[:3]

        self._set_obj_xyz(self.obj_init_pos)

        self.liftThresh = 0.04
        self.objHeight = self.data.geom("objGeom").xpos[2]
        self.heightTarget = self.objHeight + self.liftThresh

        self.maxReachDist = np.linalg.norm(self.init_tcp - np.array(self._target_pos))
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
        else:
            objPos = obs[4:7]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            heightTarget = self.heightTarget
            goal = self._target_pos

            def compute_reward_reach(actions, obs):
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

            def compute_reward_push(actions, obs):
                del actions
                del obs

                c1 = 1000
                c2 = 0.01
                c3 = 0.001
                assert np.all(goal == self._get_site_pos("goal_push"))
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
                return [
                    reward,
                    reachRew,
                    reachDist,
                    pushRew,
                    pushDist,
                    None,
                    None,
                    None,
                ]

            def compute_reward_pick_place(actions, obs):
                del obs

                reachDist = np.linalg.norm(objPos - fingerCOM)
                placingDist = np.linalg.norm(objPos - goal)
                assert np.all(goal == self._get_site_pos("goal_pick_place"))

                def reachReward():
                    reachRew = -reachDist
                    reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
                    zRew = np.linalg.norm(fingerCOM[-1] - self.init_fingerCOM[-1])

                    if reachDistxy < 0.05:
                        reachRew = -reachDist
                    else:
                        reachRew = -reachDistxy - 2 * zRew

                    if reachDist < 0.05:
                        reachRew = -reachDist + max(actions[-1], 0) / 50

                    return reachRew, reachDist

                def pickCompletionCriteria():
                    tolerance = 0.01
                    return objPos[2] >= (heightTarget - tolerance)

                self.pickCompleted = pickCompletionCriteria()

                def objDropped():
                    return (
                        (objPos[2] < (self.objHeight + 0.005))
                        and (placingDist > 0.02)
                        and (reachDist > 0.02)
                    )
                    # Object on the ground, far away from the goal, and from the gripper
                    # Can tweak the margin limits

                def orig_pickReward():
                    hScale = 100
                    if self.pickCompleted and not (objDropped()):
                        return hScale * heightTarget
                    elif (reachDist < 0.1) and (objPos[2] > (self.objHeight + 0.005)):
                        return hScale * min(heightTarget, objPos[2])
                    else:
                        return 0

                def placeReward():
                    c1 = 1000
                    c2 = 0.01
                    c3 = 0.001
                    cond = (
                        self.pickCompleted and (reachDist < 0.1) and not (objDropped())
                    )
                    if cond:
                        placeRew = 1000 * (self.maxPlacingDist - placingDist) + c1 * (
                            np.exp(-(placingDist**2) / c2)
                            + np.exp(-(placingDist**2) / c3)
                        )
                        placeRew = max(placeRew, 0)
                        return [placeRew, placingDist]
                    else:
                        return [0, placingDist]

                reachRew, reachDist = reachReward()
                pickRew = orig_pickReward()
                placeRew, placingDist = placeReward()
                assert (placeRew >= 0) and (pickRew >= 0)
                reward = reachRew + pickRew + placeRew

                return [
                    reward,
                    reachRew,
                    reachDist,
                    None,
                    None,
                    pickRew,
                    placeRew,
                    placingDist,
                ]

            return compute_reward_reach(action, obs)
