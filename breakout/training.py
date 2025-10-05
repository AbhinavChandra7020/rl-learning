import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import ale_py
import time

if __name__ == '__main__':
    log_path = os.path.join('Training', 'Logs')
    save_path = os.path.join('Training', 'Saved Models')
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    
    gym.register_envs(ale_py)

    env = make_atari_env(
        "ALE/Breakout-v5", 
        n_envs=8,
        seed=0,
        vec_env_cls=SubprocVecEnv
    )
    env = VecFrameStack(env, n_stack=4)
    
    eval_env = make_atari_env(
        "ALE/Breakout-v5",
        n_envs=1,
        seed=42
    )
    eval_env = VecFrameStack(eval_env, n_stack=4)
    
    model = DQN(
        'CnnPolicy', 
        env, 
        verbose=1, 
        tensorboard_log=log_path,
        learning_rate=1e-4,
        buffer_size=400000,
        learning_starts=100000,
        batch_size=32,
        train_freq=4,
        gradient_steps=1,
        tau=1.0,
        target_update_interval=2000,
        gamma=0.99,
        exploration_fraction=0.25,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        optimize_memory_usage=True,
        max_grad_norm=10,
        replay_buffer_kwargs=dict(handle_timeout_termination=False),
        policy_kwargs=dict(normalize_images=False),
        device='auto'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, 'best_model'),
        log_path=os.path.join(log_path, 'eval_results'),
        eval_freq=25000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=250000,
        save_path=save_path,
        name_prefix='dqn_breakout',
        verbose=1
    )
    
    print("\n" + "="*60)
    print("DQN BREAKOUT TRAINING")
    print("="*60)
    print(f"Total timesteps:      20,000,000")
    print(f"Parallel envs:        8")
    print(f"Buffer size:          400,000")
    print(f"Learning starts:      100,000")
    print(f"Target update:        every 2,000 steps")
    print(f"Exploration ends at:  5,000,000 steps (25%)")
    print(f"Device:               {model.device}")
    print(f"Eval frequency:       every 25,000 steps")
    print("="*60 + "\n")
    
    start_time = time.time()
    total_timesteps = 20_000_000
    
    model.learn(
        total_timesteps=total_timesteps, 
        tb_log_name="DQN_Breakout",
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    final_model_path = os.path.join(save_path, 'dqn_breakout_final')
    model.save(final_model_path)
    
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=20,
        deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    end_time = time.time()
    training_time = end_time - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    
    print(f"\nTraining completed in {hours}h {minutes}m")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {os.path.join(save_path, 'best_model')}")
    print("="*60 + "\n")
    
    env.close()
    eval_env.close()