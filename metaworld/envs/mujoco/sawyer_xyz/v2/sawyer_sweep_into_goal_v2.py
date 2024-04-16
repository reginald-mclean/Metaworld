import numpy as np
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerSweepIntoGoalEnvV2(SawyerXYZEnv):
    OBJ_RADIUS = 0.02

    def __init__(self, render_mode=None, reward_func_version="v2"):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.02)
        obj_high = (0.1, 0.7, 0.02)
        goal_low = (-0.001, 0.8399, 0.0199)
        goal_high = (+0.001, 0.8401, 0.0201)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
        )

        self.reward_func_version = reward_func_version

        self.init_config = {
            "obj_init_pos": np.array([0.0, 0.6, 0.02]),
            "obj_init_angle": 0.3,
            "hand_init_pos": np.array([0.0, 0.6, 0.2]),
        }
        self.goal = np.array([0.0, 0.84, 0.02])
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
        # obj = obs[4:7]
        (reward, target_to_obj) = self.compute_reward(action, obs)

        info = {"success": float(target_to_obj <= 0.05)}
        return reward, info

    def _get_quat_objects(self):
        geom_xmat = self.data.geom("objGeom").xmat.reshape(3, 3)
        return Rotation.from_matrix(geom_xmat).as_quat()

    def _get_pos_objects(self):
        return self.get_body_com("obj")

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.get_body_com("obj")
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.objHeight = self.get_body_com("obj")[2]

        goal_pos = self._get_state_rand_vec()
        while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
            goal_pos = self._get_state_rand_vec()
        self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))

        self._set_obj_xyz(self.obj_init_pos)
        self.maxPushDist = np.linalg.norm(
            self.obj_init_pos[:2] - np.array(self._target_pos)[:2]
        )

        return self._get_obs()

    def _gripper_caging_reward(self, action, obj_position, obj_radius):
        pad_success_margin = 0.05
        grip_success_margin = obj_radius + 0.005
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
            _TARGET_RADIUS = 0.05
            tcp = self.tcp_center
            obj = obs[4:7]
            tcp_opened = obs[3]
            target = np.array([self._target_pos[0], self._target_pos[1], obj[2]])

            obj_to_target = np.linalg.norm(obj - target)
            tcp_to_obj = np.linalg.norm(obj - tcp)
            in_place_margin = np.linalg.norm(self.obj_init_pos - target)

            in_place = reward_utils.tolerance(
                obj_to_target,
                bounds=(0, _TARGET_RADIUS),
                margin=in_place_margin,
                sigmoid="long_tail",
            )

            object_grasped = self._gripper_caging_reward(action, obj, self.OBJ_RADIUS)
            in_place_and_object_grasped = reward_utils.hamacher_product(
                object_grasped, in_place
            )

            reward = (2 * object_grasped) + (6 * in_place_and_object_grasped)

            if obj_to_target < _TARGET_RADIUS:
                reward = 10.0
            return [reward, obj_to_target]
        elif self.reward_func_version == 'v1':
            del action

            objPos = obs[4:7]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            goal = self._target_pos

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            assert np.all(goal == self._get_site_pos("goal"))
            reachDist = np.linalg.norm(fingerCOM - objPos)
            pushDist = np.linalg.norm(objPos[:2] - goal[:2])
            reachRew = -reachDist

            self.reachCompleted = reachDist < 0.05

            if objPos[-1] < self.obj_init_pos[-1] - 0.05 and 0.4 < objPos[1] < 1.0:
                reachRew = 0
                reachDist = 0
                pushDist = 0

            if self.reachCompleted:
                pushRew = 1000 * (self.maxPushDist - pushDist) + c1 * (
                    np.exp(-(pushDist**2) / c2) + np.exp(-(pushDist**2) / c3)
                )
                pushRew = max(pushRew, 0)
            else:
                pushRew = 0
            reward = reachRew + pushRew

            return [reward, pushDist]
        elif self.reward_func_version == 'text2reward':
            # Calculate the Euclidean distance between the end-effector and the puck
            gripper_obj_dist = np.linalg.norm(obs[:3] - obs[4:7])

            # Calculate the Euclidean distance between the puck and the goal hole
            obj_goal_dist = np.linalg.norm(obs[4:7] - obs[-3:])

            # The reward for getting the puck into the hole
            if obj_goal_dist < 0.05:  # Threshold for 'close enough'
                reward = 1.0
            else:
                # Reward is higher the closer the gripper is to the puck and the puck is to the goal
                # We want to minimize these distances, so we negate them
                # We also scale the distances by some factor to control their impact on the total reward
                reward = -0.01 * gripper_obj_dist - 0.01 * obj_goal_dist

            # Regularization of the robot's action
            # We want to encourage the robot to take smoother actions, so we penalize large actions
            action_penalty = 0.001 * np.sum(np.square(action))
            reward -= action_penalty


            obj = obs[4:7]
            target = np.array([self._target_pos[0], self._target_pos[1], obj[2]])

            obj_to_target = np.linalg.norm(obj - target)

            return reward, obj_to_target
