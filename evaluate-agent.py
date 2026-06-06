import os
 
import gymnasium as gym
import panda_gym

from huggingface_sb3 import load_from_hub, package_to_hub
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from huggingface_hub import notebook_login


#load saved stats

eval_env = DummyVecEnv([lambda: gym.make("PandaReachDense-v3")])
eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)

#override render mode
eval_env.render_mode = "rgb_array"

#do not train at test time
eval_env.training = False
#when evaluating, reward normalization not needed
eval_env.norm_reward = False

#load agent / model
model = A2C.load("a2c-PandaReachDense-v3")

#evaluate policy to obtain mean reward and std of reward
mean_reward, std_reward = evaluate_policy(model, eval_env)

#print results
print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")