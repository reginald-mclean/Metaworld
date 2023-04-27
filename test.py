import metaworld
import random


random.seed(42)
print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

# push, reach, sweep

ml1 = metaworld.ML1('sweep-v2') # Construct the benchmark, sampling tasks

env = ml1.train_classes['sweep-v2']()  # Create an environment with task `pick_place`
task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task

obs = env.reset()  # Reset environment
# print(obs)
# a = env.action_space.sample()  # Sample an action
# obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action