import numpy as np
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerPushBackEnvV2(SawyerXYZEnv):
    OBJ_RADIUS = 0.007
    TARGET_RADIUS = 0.05

    def __init__(self, render_mode=None, reward_func_version="v2"):
        goal_low = (-0.1, 0.6, 0.0199)
        goal_high = (0.1, 0.7, 0.0201)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.8, 0.02)
        obj_high = (0.1, 0.85, 0.02)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
        )

        self.reward_func_version = reward_func_version

        self.init_config = {
            "obj_init_pos": np.array([0, 0.8, 0.02]),
            "obj_init_angle": 0.3,
            "hand_init_pos": np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0.0, 0.6, 0.02])
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
        return full_v2_path_for("sawyer_xyz/sawyer_push_back_v2.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (reward, target_to_obj) = self.compute_reward(action, obs)

        success = float(target_to_obj <= 0.07)
        info = {"success": success}
        return reward, info

    def _get_pos_objects(self):
        return self.data.geom("objGeom").xpos

    def _get_quat_objects(self):
        return Rotation.from_matrix(
            self.data.geom("objGeom").xmat.reshape(3, 3)
        ).as_quat()

    def adjust_initObjPos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not aligned
        # If this is not done, the object could be initialized in an extreme position
        diff = self.get_body_com("obj")[:2] - self.data.geom("objGeom").xpos[:2]
        adjustedPos = orig_init_pos[:2] + diff

        # The convention we follow is that body_com[2] is always 0, and geom_pos[2] is the object height
        return [adjustedPos[0], adjustedPos[1], self.data.geom("objGeom").xpos[-1]]

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.adjust_initObjPos(self.init_config["obj_init_pos"])
        self.obj_init_angle = self.init_config["obj_init_angle"]

        goal_pos = self._get_state_rand_vec()
        self._target_pos = np.concatenate((goal_pos[-3:-1], [self.obj_init_pos[-1]]))
        while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = np.concatenate(
                (goal_pos[-3:-1], [self.obj_init_pos[-1]])
            )
        self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))

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

    def _gripper_caging_reward(self, action, obj_position, obj_radius):
        pad_success_margin = 0.05
        grip_success_margin = obj_radius + 0.003
        x_z_success_margin = 0.01

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

        right_gripping = reward_utils.tolerance(
            delta_object_y_right_pad,
            bounds=(obj_radius, grip_success_margin),
            margin=right_caging_margin,
            sigmoid="long_tail",
        )
        left_gripping = reward_utils.tolerance(
            delta_object_y_left_pad,
            bounds=(obj_radius, grip_success_margin),
            margin=left_caging_margin,
            sigmoid="long_tail",
        )

        assert right_caging >= 0 and right_caging <= 1
        assert left_caging >= 0 and left_caging <= 1

        y_caging = reward_utils.hamacher_product(right_caging, left_caging)
        y_gripping = reward_utils.hamacher_product(right_gripping, left_gripping)

        assert y_caging >= 0 and y_caging <= 1

        tcp_xz = tcp + np.array([0.0, -tcp[1], 0.0])
        obj_position_x_z = np.copy(obj_position) + np.array(
            [0.0, -obj_position[1], 0.0]
        )
        tcp_obj_norm_x_z = np.linalg.norm(tcp_xz - obj_position_x_z, ord=2)
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

        assert right_caging >= 0 and right_caging <= 1
        gripper_closed = min(max(0, action[-1]), 1)
        assert gripper_closed >= 0 and gripper_closed <= 1
        caging = reward_utils.hamacher_product(y_caging, x_z_caging)
        assert caging >= 0 and caging <= 1

        if caging > 0.95:
            gripping = y_gripping
        else:
            gripping = 0.0
        assert gripping >= 0 and gripping <= 1

        caging_and_gripping = (caging + gripping) / 2
        assert caging_and_gripping >= 0 and caging_and_gripping <= 1

        return caging_and_gripping

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
            object_grasped = self._gripper_caging_reward(action, obj, self.OBJ_RADIUS)

            reward = reward_utils.hamacher_product(object_grasped, in_place)

            if (
                (tcp_to_obj < 0.01)
                and (0 < tcp_opened < 0.55)
                and (target_to_obj_init - target_to_obj > 0.01)
            ):
                reward += 1.0 + 5.0 * in_place
            if target_to_obj < self.TARGET_RADIUS:
                reward = 10.0
            return (reward, target_to_obj)
        else:
            objPos = obs[4:7]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            heightTarget = self.heightTarget
            goal = self._target_pos

            def compute_reward_push(actions, obs):
                c1 = 1000
                c2 = 0.01
                c3 = 0.001
                del actions
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

                return [reward, placingDist]

            return compute_reward_push(action, obs)
