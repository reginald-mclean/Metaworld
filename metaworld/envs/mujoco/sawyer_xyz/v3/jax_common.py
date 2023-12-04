from datetime import datetime
import functools
import jax
from jax import numpy as jp
import numpy as np
from typing import Any, Dict, Tuple, Union

from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, State
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from etils import epath
from flax import struct
from ml_collections import config_dict
import mujoco
from mujoco import mjx

@struct.dataclass
class State(Base):
    """Environment state for training and inference with brax.

    Args:
      pipeline_state: the physics state, mjx.Data
      obs: environment observations
      reward: environment reward
      done: boolean, True if the current episode has terminated
      metrics: metrics that get tracked per environment step
      info: environment variables defined and updated by the environment reset
      and step functions
    """

    pipeline_state: mjx.Data
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)

class MjxEnv(Env):
    """API for driving an MJX system for training and inference in brax."""
    def __init__(
        self,
        mj_model: mujoco.MjModel,
        physics_steps_per_control_step: int = 1,
    ):
        """Initializes MjxEnv.

        Args:
         mj_model: mujoco.MjModel
         physics_steps_per_control_step: the number of times to step the physics
          pipeline for each environment step
        """
        self.model = mujoco.MjModel.from_xml_path(mj_model)
        self.data = mujoco.MjData(self.model)
        self.sys = mjx.device_put(self.model)
        self._physics_steps_per_control_step = physics_steps_per_control_step

    def pipeline_init(self) -> mjx.Data:
        """Initializes the physics state."""
        data = mjx.device_put(self.data)
        data = mjx.forward(self.sys, data)
        return data

    def pipeline_step(
      self, data: mjx.Data, ctrl: jax.Array
    ) -> mjx.Data:
        """Takes a physics step using the physics pipeline."""
        def f(data, _):
          data = data.replace(ctrl=ctrl)
          return (
              mjx.step(self.sys, data),
              None,
          )
        data, _ = jax.lax.scan(f, data, (), self._physics_steps_per_control_step)
        return data

    @property
    def dt(self) -> jax.Array:
        """The timestep used for each env step."""
        return self.sys.opt.timestep * self._physics_steps_per_control_step

    @property
    def observation_size(self) -> int:
        rng = jax.random.PRNGKey(0)
        reset_state = self.unwrapped.reset(rng)
        return reset_state.obs.shape[-1]

    @property
    def action_size(self) -> int:
        return self.sys.nu

    @property
    def backend(self) -> str:
        return 'mjx'

    def _pos_vel(self, data: mjx.Data) -> Tuple[Transform, Motion]:
        """Returns 6d spatial transform and 6d velocity for all bodies."""
        x = Transform(pos=data.xpos[1:, :], rot=data.xquat[1:, :])
        cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
        offset = data.xpos[1:, :] - data.subtree_com[
            self.model.body_rootid[np.arange(1, self.model.nbody)]]
        xd = Transform.create(pos=offset).vmap().do(cvel)
        return x, xd

    def reset(self, rng:jp.array) -> State:
        data = self.pipeline_init()
        self.reset_model(data)
        obs = self._get_obs()
        return State(obs)

    def step(state: State, action: jp.array) -> State:
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)
        return data


    def _get_pos_goal(self):
        """Retrieves goal position from mujoco properties or instance vars.

        Returns:
            np.ndarray: Flat array (3 elements) representing the goal position
        """
        assert isinstance(self._target_pos, np.ndarray)
        assert self._target_pos.ndim == 1
        return self._target_pos

    def _get_obs(self) -> jp.array:
        pos_goal = self._get_pos_goal()
        if self._partially_observable:
            pos_goal = np.zeros_like(pos_goal)
        curr_obs = self._get_curr_obs_combined_no_goal()
        # do frame stacking
        obs = np.hstack((curr_obs, self._prev_obs, pos_goal))
        self._prev_obs = curr_obs
        return jp.array(obs)

    def _reset_hand(self, data, steps=5) -> None:
        mocap_id = self.model.body_mocapid[self.data.body("mocap").id]
        for _ in range(steps):
            data.mocap_pos[mocap_id][:] = self.hand_init_pos
            data.mocap_quat[mocap_id][:] = np.array([1, 0, 1, 0])
            data.ctrl[:] = [-1, 1]
            mujoco.mj_step(self.sys, data, nstep=5)
        self.init_tcp = self.tcp_center
