import os
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import ale_py
import time

if __name__ == '__main__':  
    log_path = os.path.join('Training', 'Logs')
    a2c_path = os.path.join('Training', 'Saved Models', 'A2C_Breakout_Model')

    gym.register_envs(ale_py)

    # Use SubprocVecEnv for multi-core processing
    env = make_atari_env("ALE/Breakout-v5", 
                         n_envs=32, 
                         seed=1,
                         vec_env_cls=SubprocVecEnv)
    
    env = VecFrameStack(env, n_stack=4)

    model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=log_path)

    start_time = time.time()

    model.learn(total_timesteps=2000000)

    end_time = time.time()
    training_time = end_time - start_time

    minutes = int(training_time // 60)
    seconds = int(training_time % 60)
    print(f"\n{'='*50}")
    print(f"Training completed in {minutes} minutes and {seconds} seconds")
    print(f"Total time: {training_time:.2f} seconds")
    print(f"{'='*50}\n")

    model.save(a2c_path)
    
    env.close()