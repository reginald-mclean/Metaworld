import numpy as np
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerPickPlaceEnvV2(SawyerXYZEnv):
    """SawyerPickPlaceEnv.

    Motivation for V2:
        V1 was very difficult to solve because the observation didn't say where
        to move after picking up the puck.
    Changelog from V1 to V2:
        - (7/7/20) Removed 3 element vector. Replaced with 3 element position
            of the goal (for consistency with other environments)
        - (6/15/20) Added a 3 element vector to the observation. This vector
            points from the end effector to the goal coordinate.
            i.e. (self._target_pos - pos_hand)
        - (6/15/20) Separated reach-push-pick-place into 3 separate envs.
    """

    def __init__(self, tasks=None, render_mode=None, reward_func_version='v2'):
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
        )

        self.reward_func_version = reward_func_version

        self.init_config = {
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0, 0.6, 0.02]),
            "hand_init_pos": np.array([0, 0.6, 0.2]),
        }

        self.goal = np.array([0.1, 0.8, 0.2])

        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.num_resets = 0
        self.obj_init_pos = None

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_pick_place_v2.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        obj = obs[4:7]

        (
            reward,
            obj_to_target
        ) = self.compute_reward(action, obs)
        success = float(obj_to_target <= 0.07)
        info = {
            "success": success,
        }

        return reward, info

    @property
    def _get_id_main_object(self):
        return self.data.geom("objGeom").id

    def _get_pos_objects(self):
        return self.get_body_com("obj")

    def _get_quat_objects(self):
        return Rotation.from_matrix(
            self.data.geom("objGeom").xmat.reshape(3, 3)
        ).as_quat()

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
        self.obj_init_pos = self.fix_extreme_obj_pos(self.init_config["obj_init_pos"])
        self.obj_init_angle = self.init_config["obj_init_angle"]

        goal_pos = self._get_state_rand_vec()
        self._target_pos = goal_pos[3:]
        while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = goal_pos[3:]
        self._target_pos = goal_pos[-3:]
        self.obj_init_pos = goal_pos[:3]
        self.init_tcp = self.tcp_center
        self.init_left_pad = self.get_body_com("leftpad")
        self.init_right_pad = self.get_body_com("rightpad")

        self._set_obj_xyz(self.obj_init_pos)

        self.objHeight = self.data.geom("objGeom").xpos[2]
        self.heightTarget = self.objHeight + 0.04

        self.maxPlacingDist = (
                np.linalg.norm(
                    np.array(
                        [self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]
                    )
                    - np.array(self._target_pos)
                )
                + self.heightTarget
        )

        self.maxPushDist = np.linalg.norm(
            self.obj_init_pos[:2] - np.array(self._target_pos)[:2]
        )
        return self._get_obs()

    def _gripper_caging_reward(self, action, obj_position):
        pad_success_margin = 0.05
        x_z_success_margin = 0.005
        obj_radius = 0.015
        tcp = self.tcp_center
        left_pad = self.get_body_com("leftpad")
        right_pad = self.get_body_com("rightpad")
        delta_object_y_left_pad = left_pad[1] - obj_position[1]
        delta_object_y_right_pad = obj_position[1] - right_pad[1]
        right_caging_margin = abs(
            abs(obj_position[1] - self.init_right_pad[1]) - pad_success_margin
        )
        left_caging_margin = abs(
            abs(obj_position[1] - self.init_left_pad[1]) - pad_success_margin
        )

        right_caging = reward_utils.tolerance(
            delta_object_y_right_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=right_caging_margin,
            sigmoid="long_tail",
        )
        left_caging = reward_utils.tolerance(
            delta_object_y_left_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=left_caging_margin,
            sigmoid="long_tail",
        )

        y_caging = reward_utils.hamacher_product(left_caging, right_caging)

        # compute the tcp_obj distance in the x_z plane
        tcp_xz = tcp + np.array([0.0, -tcp[1], 0.0])
        obj_position_x_z = np.copy(obj_position) + np.array(
            [0.0, -obj_position[1], 0.0]
        )
        tcp_obj_norm_x_z = np.linalg.norm(tcp_xz - obj_position_x_z, ord=2)

        # used for computing the tcp to object object margin in the x_z plane
        init_obj_x_z = self.obj_init_pos + np.array([0.0, -self.obj_init_pos[1], 0.0])
        init_tcp_x_z = self.init_tcp + np.array([0.0, -self.init_tcp[1], 0.0])
        tcp_obj_x_z_margin = (
            np.linalg.norm(init_obj_x_z - init_tcp_x_z, ord=2) - x_z_success_margin
        )

        x_z_caging = reward_utils.tolerance(
            tcp_obj_norm_x_z,
            bounds=(0, x_z_success_margin),
            margin=tcp_obj_x_z_margin,
            sigmoid="long_tail",
        )

        gripper_closed = min(max(0, action[-1]), 1)
        caging = reward_utils.hamacher_product(y_caging, x_z_caging)

        gripping = gripper_closed if caging > 0.97 else 0.0
        caging_and_gripping = reward_utils.hamacher_product(caging, gripping)
        caging_and_gripping = (caging_and_gripping + caging) / 2
        return caging_and_gripping

    def compute_reward(self, action, obs):
        if self.reward_func_version == 'v2':
            _TARGET_RADIUS = 0.05
            tcp = self.tcp_center
            obj = obs[4:7]
            tcp_opened = obs[3]
            target = self._target_pos

            obj_to_target = np.linalg.norm(obj - target)
            tcp_to_obj = np.linalg.norm(obj - tcp)
            in_place_margin = np.linalg.norm(self.obj_init_pos - target)

            in_place = reward_utils.tolerance(
                obj_to_target,
                bounds=(0, _TARGET_RADIUS),
                margin=in_place_margin,
                sigmoid="long_tail",
            )

            object_grasped = self._gripper_caging_reward(action, obj)
            in_place_and_object_grasped = reward_utils.hamacher_product(
                object_grasped, in_place
            )
            reward = in_place_and_object_grasped

            if (
                tcp_to_obj < 0.02
                and (tcp_opened > 0)
                and (obj[2] - 0.01 > self.obj_init_pos[2])
            ):
                reward += 1.0 + 5.0 * in_place
            if obj_to_target < _TARGET_RADIUS:
                reward = 10.0
            return [reward, tcp_to_obj, tcp_opened, obj_to_target, object_grasped, in_place]
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
                        np.exp(-(reachDist ** 2) / c2) + np.exp(-(reachDist ** 2) / c3)
                )
                reachRew = max(reachRew, 0)
                reward = reachRew
                return [reward, reachRew, reachDist, None, None, None, None, None]

            def compute_reward_push(actions, obs):
                c1 = 1000
                c2 = 0.01
                c3 = 0.001
                del actions
                del obs

                assert np.all(goal == self._get_site_pos("goal_push"))
                reachDist = np.linalg.norm(fingerCOM - objPos)
                pushDist = np.linalg.norm(objPos[:2] - goal[:2])
                reachRew = -reachDist
                if reachDist < 0.05:
                    pushRew = 1000 * (self.maxPushDist - pushDist) + c1 * (
                            np.exp(-(pushDist ** 2) / c2) + np.exp(-(pushDist ** 2) / c3)
                    )
                    pushRew = max(pushRew, 0)
                else:
                    pushRew = 0
                reward = reachRew + pushRew
                return [reward, reachRew, reachDist, pushRew, pushDist, None, None, None]

            def compute_reward_pick_place(actions, obs):
                del obs

                reachDist = np.linalg.norm(objPos - fingerCOM)
                placingDist = np.linalg.norm(objPos - goal)
                assert np.all(goal == self._get_site_pos("goal"))

                def reachReward():
                    reachRew = -reachDist
                    reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
                    zRew = np.linalg.norm(fingerCOM[-1] - self.init_tcp[-1])

                    if reachDistxy < 0.05:
                        reachRew = -reachDist
                    else:
                        reachRew = -reachDistxy - 2 * zRew

                    # incentive to close fingers when reachDist is small
                    if reachDist < 0.05:
                        reachRew = -reachDist + max(actions[-1], 0) / 50

                    return reachRew, reachDist

                def pickCompletionCriteria():
                    tolerance = 0.01
                    if objPos[2] >= (heightTarget - tolerance):
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
                    cond = self.pickCompleted and (reachDist < 0.1) and not (objDropped())
                    if cond:
                        placeRew = 1000 * (self.maxPlacingDist - placingDist) + c1 * (
                                np.exp(-(placingDist ** 2) / c2)
                                + np.exp(-(placingDist ** 2) / c3)
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
                    placingDist
                ]


            return compute_reward_pick_place(action, obs)

