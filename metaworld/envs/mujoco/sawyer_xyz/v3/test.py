import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"

from jax.config import config

config.update('jax_disable_jit', True)

from sawyer_assembly_peg_v2 import SawyerNutAssemblyEnvV2
import jax


env = SawyerNutAssemblyEnvV2()
env._freeze_rand_vec = False
env._set_task_called = True



print(env.reset(jax.random.PRNGKey(0)))
