import mujoco
import numpy as np
from gymnasium.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerHandlePressSideEnvV2(SawyerXYZEnv):
    """SawyerHandlePressSideEnv.

    Motivation for V2:
        V1 was very difficult to solve because the end effector's wrist has a
        nub that got caught on the box before pushing the handle all the way
        down. There are a number of ways to fix this, e.g. moving box to right
        side of table, extending handle's length, decreasing handle's damping,
        or moving the goal position slightly upward. I just the last one.
    Changelog from V1 to V2:
        - (8/05/20) Updated to new XML
        - (6/30/20) Increased goal's Z coordinate by 0.01 in XML
    """

    TARGET_RADIUS = 0.02

    def __init__(self, render_mode=None, reward_func_version="v2"):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1.0, 0.5)
        obj_low = (-0.35, 0.65, -0.001)
        obj_high = (-0.25, 0.75, 0.001)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
        )

        self.reward_func_version = reward_func_version

        self.init_config = {
            "obj_init_pos": np.array([-0.3, 0.7, 0.0]),
            "hand_init_pos": np.array(
                (0, 0.6, 0.2),
            ),
        }
        self.goal = np.array([-0.2, 0.7, 0.14])
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
        return full_v2_path_for("sawyer_xyz/sawyer_handle_press_sideways.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (reward, target_to_obj) = self.compute_reward(action, obs)

        info = {"success": float(target_to_obj <= self.TARGET_RADIUS)}

        return reward, info

    @property
    def _target_site_config(self):
        return []

    def _get_pos_objects(self):
        return self._get_site_pos("handleStart")

    def _get_quat_objects(self):
        return np.zeros(4)

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()

        self.obj_init_pos = self._get_state_rand_vec()

        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        ] = self.obj_init_pos
        self._set_obj_xyz(-0.001)
        self._target_pos = self._get_site_pos("goalPress")
        self._handle_init_pos = self._get_pos_objects()

        self.maxDist = np.abs(
            self.data.site("handleStart").xpos[-1] - self._target_pos[-1]
        )

        return self._get_obs()

    def compute_reward(self, actions, obs):
        if self.reward_func_version == "v2":
            del actions
            obj = self._get_pos_objects()
            tcp = self.tcp_center
            target = self._target_pos.copy()

            target_to_obj = obj[2] - target[2]
            target_to_obj = np.linalg.norm(target_to_obj)
            target_to_obj_init = self._handle_init_pos[2] - target[2]
            target_to_obj_init = np.linalg.norm(target_to_obj_init)

            in_place = reward_utils.tolerance(
                target_to_obj,
                bounds=(0, self.TARGET_RADIUS),
                margin=abs(target_to_obj_init - self.TARGET_RADIUS),
                sigmoid="long_tail",
            )

            handle_radius = 0.02
            tcp_to_obj = np.linalg.norm(obj - tcp)
            tcp_to_obj_init = np.linalg.norm(self._handle_init_pos - self.init_tcp)
            reach = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, handle_radius),
                margin=abs(tcp_to_obj_init - handle_radius),
                sigmoid="long_tail",
            )
            tcp_opened = 0
            object_grasped = reach

            reward = reward_utils.hamacher_product(reach, in_place)
            reward = 1 if target_to_obj <= self.TARGET_RADIUS else reward
            reward *= 10
            return (reward, target_to_obj)
        else:
            del actions

            objPos = obs[4:7]

            leftFinger = self._get_site_pos("leftEndEffector")
            fingerCOM = leftFinger

            pressGoal = self._target_pos[-1]

            pressDist = np.abs(objPos[-1] - pressGoal)
            reachDist = np.linalg.norm(objPos - fingerCOM)

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            if reachDist < 0.05:
                pressRew = 1000 * (self.maxDist - pressDist) + c1 * (
                    np.exp(-(pressDist**2) / c2) + np.exp(-(pressDist**2) / c3)
                )
            else:
                pressRew = 0

            pressRew = max(pressRew, 0)
            reward = -reachDist + pressRew

            return [reward, pressDist]
        elif self.reward_func_version == 'text2reward':
            # Constants to weight various components of the reward
            distance_weight = 1.0
            state_difference_weight = 1.0
            action_regularization_weight = 0.1

            # Computing the Euclidean distance between the gripper and the handle
            gripper_handle_distance = np.linalg.norm(obs[:3] - obs[4:7])
            # Normalizing the distance to be within the range -1 to 1
            gripper_handle_distance = np.clip(gripper_handle_distance, -1, 1)

            # Computing the difference between the current and goal states of the handle
            state_difference = np.linalg.norm(obs[4:7] - obs[-3:])
            # Normalizing the difference to be within the range -1 to 1
            state_difference = np.clip(state_difference, -1, 1)

            # Computing the regularization of the robot's action
            action_regularization = np.linalg.norm(actions)
            # Normalizing the regularization to be within the range -1 to 1
            action_regularization = np.clip(action_regularization, -1, 1)

            # Weighting and summing the components to compute the final reward
            reward = -distance_weight * gripper_handle_distance \
              -state_difference_weight * state_difference \
              -action_regularization_weight * action_regularization

            obj = self._get_pos_objects()
            target = self._target_pos.copy()

            target_to_obj = obj[2] - target[2]
            target_to_obj = np.linalg.norm(target_to_obj)

            return reward, target_to_obj
