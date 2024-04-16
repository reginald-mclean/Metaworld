import numpy as np
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerDoorCloseEnvV2(SawyerXYZEnv):
    def __init__(self, render_mode=None, reward_func_version="v2"):
        goal_low = (0.2, 0.65, 0.1499)
        goal_high = (0.3, 0.75, 0.1501)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0.0, 0.85, 0.15)
        obj_high = (0.1, 0.95, 0.15)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
        )

        self.reward_func_version = reward_func_version

        self.init_config = {
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0.1, 0.95, 0.15], dtype=np.float32),
            "hand_init_pos": np.array([-0.5, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0.2, 0.8, 0.15])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self.door_qpos_adr = self.model.joint("doorjoint").qposadr.item()
        self.door_qvel_adr = self.model.joint("doorjoint").dofadr.item()

        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_door_pull.xml")

    def _get_pos_objects(self):
        return self.data.geom("handle").xpos.copy()

    def _get_quat_objects(self):
        return Rotation.from_matrix(
            self.data.geom("handle").xmat.reshape(3, 3)
        ).as_quat()

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[self.door_qpos_adr] = pos
        qvel[self.door_qvel_adr] = 0
        self.set_state(qpos.flatten(), qvel.flatten())

    def reset_model(self):
        self._reset_hand()
        self.objHeight = self.data.geom("handle").xpos[2]
        obj_pos = self._get_state_rand_vec()
        self.obj_init_pos = obj_pos
        goal_pos = obj_pos.copy() + np.array([0.2, -0.2, 0.0])
        self._target_pos = goal_pos

        self.model.body("door").pos = self.obj_init_pos
        self.model.site("goal").pos = self._target_pos

        # keep the door open after resetting initial positions
        self._set_obj_xyz(-1.5708)

        self.maxPullDist = np.linalg.norm(
            self.data.geom("handle").xpos[:-1] - self._target_pos[:-1]
        )

        return self._get_obs()

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        reward, obj_to_target = self.compute_reward(action, obs)
        info = {"success": float(obj_to_target <= 0.08)}
        return reward, info

    def compute_reward(self, actions, obs):
        if self.reward_func_version == "v2":
            _TARGET_RADIUS = 0.05
            tcp = self.tcp_center
            obj = obs[4:7]
            target = self._target_pos

            tcp_to_target = np.linalg.norm(tcp - target)
            # tcp_to_obj = np.linalg.norm(tcp - obj)
            obj_to_target = np.linalg.norm(obj - target)

            in_place_margin = np.linalg.norm(self.obj_init_pos - target)
            in_place = reward_utils.tolerance(
                obj_to_target,
                bounds=(0, _TARGET_RADIUS),
                margin=in_place_margin,
                sigmoid="gaussian",
            )

            hand_margin = np.linalg.norm(self.hand_init_pos - obj) + 0.1
            hand_in_place = reward_utils.tolerance(
                tcp_to_target,
                bounds=(0, 0.25 * _TARGET_RADIUS),
                margin=hand_margin,
                sigmoid="gaussian",
            )

            reward = 3 * hand_in_place + 6 * in_place

            if obj_to_target < _TARGET_RADIUS:
                reward = 10

            return [reward, obj_to_target]
        elif self.reward_func_version == 'v1':
            del actions
            objPos = obs[4:7]

            rightFinger, leftFinger = self._get_site_pos(
                "rightEndEffector"
            ), self._get_site_pos("leftEndEffector")
            fingerCOM = (rightFinger + leftFinger) / 2

            pullGoal = self._target_pos

            pullDist = np.linalg.norm(objPos[:-1] - pullGoal[:-1])
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
                pullRew = pullRew
            else:
                pullRew = 0

            reward = reachRew + pullRew

            return [reward, pullDist]
        elif self.reward_func_version == 'text2reward':
            # Calculate the distance between the end-effector and the door handle
            distance_to_handle = np.linalg.norm(obs[:3] - obs[4:7])

            # Calculate the distance between the door handle's current position and the goal position
            distance_to_goal = np.linalg.norm(obs[4:7] - obs[-3:])

            # Reward for reaching the door handle
            reach_reward = -distance_to_handle

            # Reward for pushing the door handle towards the goal position
            push_reward = -distance_to_goal

            # Encourage the gripper to close when near the door handle
            gripper_reward = 0
            if distance_to_handle < 0.1:
                gripper_reward = -obs[3]

            # Combine the rewards
            reward = reach_reward + push_reward + gripper_reward

            obj = obs[4:7]
            target = self._target_pos

            obj_to_target = np.linalg.norm(obj - target)

            return reward, obj_to_target

