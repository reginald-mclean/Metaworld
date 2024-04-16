import numpy as np
from gymnasium.spaces import Box
from scipy.spatial.distance import cdist
from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerDoorUnlockEnvV2(SawyerXYZEnv):
    def __init__(self, render_mode=None, reward_func_version="v2"):
        hand_low = (-0.5, 0.40, -0.15)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.8, 0.15)
        obj_high = (0.1, 0.85, 0.15)
        goal_low = (0.0, 0.64, 0.2100)
        goal_high = (0.2, 0.7, 0.2111)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
        )

        self.reward_func_version = reward_func_version

        self.init_config = {
            "obj_init_pos": np.array([0, 0.85, 0.15]),
            "hand_init_pos": np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0, 0.85, 0.1])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._lock_length = 0.1

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_door_lock.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (reward, obj_to_target) = self.compute_reward(action, obs)

        info = {"success": float(obj_to_target <= 0.02)}

        return reward, info

    @property
    def _target_site_config(self):
        return [
            ("goal_unlock", self._target_pos),
            ("goal_lock", np.array([10.0, 10.0, 10.0])),
        ]

    def _get_id_main_object(self):
        return None

    def _get_pos_objects(self):
        return self._get_site_pos("lockStartUnlock")

    def _get_quat_objects(self):
        return self.data.body("door_link").xquat

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        self.model.body("door").pos = self._get_state_rand_vec()
        self._set_obj_xyz(1.5708)

        self.obj_init_pos = self.data.body("lock_link").xpos
        self._target_pos = self.obj_init_pos + np.array([0.1, -0.04, 0.0])

        self.maxPullDist = np.linalg.norm(self._target_pos - self.obj_init_pos)

        return self._get_obs()

    def compute_reward(self, action, obs):
        if self.reward_func_version == "v2":
            del action
            gripper = obs[:3]
            lock = obs[4:7]

            # Add offset to track gripper's shoulder, rather than fingers
            offset = np.array([0.0, 0.055, 0.07])

            scale = np.array([0.25, 1.0, 0.5])
            shoulder_to_lock = (gripper + offset - lock) * scale
            shoulder_to_lock_init = (self.init_tcp + offset - self.obj_init_pos) * scale

            # This `ready_to_push` reward should be a *hint* for the agent, not an
            # end in itself. Make sure to devalue it compared to the value of
            # actually unlocking the lock
            ready_to_push = reward_utils.tolerance(
                np.linalg.norm(shoulder_to_lock),
                bounds=(0, 0.02),
                margin=np.linalg.norm(shoulder_to_lock_init),
                sigmoid="long_tail",
            )

            obj_to_target = abs(self._target_pos[0] - lock[0])
            pushed = reward_utils.tolerance(
                obj_to_target,
                bounds=(0, 0.005),
                margin=self._lock_length,
                sigmoid="long_tail",
            )

            reward = 2 * ready_to_push + 8 * pushed

            return (reward, obj_to_target)
        elif self.reward_func_version == 'v1':
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

            c1 = 1000
            c2 = 0.01
            c3 = 0.001

            if self.reachCompleted:
                pullRew = 1000 * (self.maxPullDist - pullDist) + c1 * (
                    np.exp(-(pullDist**2) / c2) + np.exp(-(pullDist**2) / c3)
                )
                pullRew = max(pullRew, 0)
            else:
                pullRew = 0

            reward = reachRew + pullRew

            return [reward, pullDist]
        elif self.reward_func_version == 'text2reward':
            # Define constants for reward tuning
            DISTANCE_WEIGHT = 1.0
            GOAL_REACHED_REWARD = 100.0
            ACTION_PENALTY = 0.1

            # Compute distance between robot's gripper and the lock
            distance = np.linalg.norm(obs[:3] - obs[4:7])

            # Compute difference between current state of object and its goal state
            goal_diff = np.linalg.norm(obs[4:7] - obs[-3:])

            # Compute action regularization term
            action_penalty = ACTION_PENALTY * np.square(action).sum()

            # Check if the goal has been reached
            goal_reached = cdist(obs[4:7].reshape(1, -1), obs[-3:].reshape(1, -1), 'cosine') < 0.01

            # Calculate reward
            reward = - DISTANCE_WEIGHT * distance - goal_diff - action_penalty
            if goal_reached:
                reward += GOAL_REACHED_REWARD


            lock = obs[4:7]
            obj_to_target = abs(self._target_pos[0] - lock[0])
            return reward, obj_to_target
