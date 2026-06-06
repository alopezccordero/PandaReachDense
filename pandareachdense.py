import os
 
import gymnasium as gym
import panda_gym

from huggingface_sb3 import load_from_hub, package_to_hub
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from huggingface_hub import notebook_login

env_id = "PandaReachDense-v3"
env = gym.make(env_id)

s_size = env.observation_space.shape
a_size = env.action_space

print("OBSERVATION SPACE:\n")
print(f"THE STATE SPACE IS: {s_size}")
print(f"Sample observation: {env.observation_space.sample()}")

print("ACTION SPACE")
print(f"The action space is: {a_size}")
print(f"Action Space Sample: {env.action_space.sample()}")

#rewards normalization

env = make_vec_env(env_id, n_envs=4)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

model = A2C(policy="MultiInputPolicy",
            env = env, 
            verbose=1)

model.learn(1_000_000)
model.save("a2c-PandaReachDense-v3")
env.save("vec_normalize.pkl")