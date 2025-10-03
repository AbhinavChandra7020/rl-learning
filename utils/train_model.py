import os
import gymnasium as gym
from stable_baselines3 import PPO 
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

timesteps = 100000

log_path = os.path.join('Training', 'Logs')
PPO_PATH = os.path.join('Training' , 'Saved Models', 'PPO_Model_CartPole')
save_path = os.path.join('Training', 'Saved Models')  

env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])

net_arch = [dict(pi = [128, 128, 128, 128], vf = [128, 128, 128, 128])]

model = PPO('MlpPolicy', env, verbose = 1, policy_kwargs = {'net_arch': net_arch})

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
eval_callback = EvalCallback(env,               callback_on_new_best=stop_callback, 
                           eval_freq=10000, 
                           best_model_save_path=save_path,
                           verbose=1)

model.learn(total_timesteps=timesteps, callback=eval_callback)

model.save(PPO_PATH)