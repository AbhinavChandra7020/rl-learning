import os
import gymnasium as gym
from stable_baselines3 import PPO 

PPO_PATH = os.path.join('Training', 'Saved Models', 'PPO_Model_CartPole')

env = gym.make('CartPole-v1', render_mode="human")
model = PPO.load(PPO_PATH)

episodes = 5
max_steps_per_episode = 500  # Set your own limit

for episode in range(1, episodes+1):
    obs, info = env.reset()  
    done = False
    score = 0
    steps = 0

    while not done and steps < max_steps_per_episode:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        score += reward
        steps += 1
        
        # Print progress every 100 steps
        if steps % 100 == 0:
            print(f'Episode {episode}, Steps: {steps}')

    # Show why episode ended
    if steps >= max_steps_per_episode:
        print('Episode:{} Score:{} Steps:{} (Reached step limit)'.format(episode, score, steps))
    else:
        print('Episode:{} Score:{} Steps:{} (Pole fell)'.format(episode, score, steps))

env.close()