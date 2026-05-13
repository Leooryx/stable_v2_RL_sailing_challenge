import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath('../src'))
sys.path.append(os.path.abspath('..'))

from agents.base_agent import BaseAgent
from env_sailing import SailingEnv
from wind_scenarios import get_wind_scenario


class MyAgent(BaseAgent):
    def __init__(self, lr=0.1, gamma=0.995, epsilon=0.9):
        super().__init__()
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.np_random = np.random.default_rng()

    def discretize_state(self, obs):
        x, y = int(obs[0]) // 8, int(obs[1]) // 8   # 16x16 position grid
        wx, wy = obs[4], obs[5]
        wind_angle = int(np.degrees(np.arctan2(wy, wx)) // 45) % 8  # 8 wind directions
        wind_strength = min(int(np.sqrt(wx**2 + wy**2) // 3), 4)    # 5 strength buckets
        return (x, y, wind_angle, wind_strength)

    def act(self, obs):
        state = self.discretize_state(obs)
        if self.np_random.random() < self.epsilon:
            return int(self.np_random.integers(0, 9))
        q = self.q_table.get(state, np.zeros(9))
        return int(np.argmax(q))

    def learn(self, state, action, reward, next_obs):
        next_state = self.discretize_state(next_obs)
        q = self.q_table.get(state, np.zeros(9)).copy()
        next_q = self.q_table.get(next_state, np.zeros(9))
        q[action] += self.lr * (reward + self.gamma * np.max(next_q) - q[action])
        self.q_table[state] = q

    def reset(self): pass
    def seed(self, seed=None): self.np_random = np.random.default_rng(seed)


agent = MyAgent()
env = SailingEnv(**get_wind_scenario('training_1'))
goal = env.goal_position

num_episodes = 2000

scenarios = ['training_1', 'training_2']

for episode in tqdm(range(num_episodes)):
    scenario = scenarios[episode % 2]
    env = SailingEnv(**get_wind_scenario(scenario))
    obs, info = env.reset(seed=episode)
    prev_dist = np.linalg.norm(info['position'] - goal)
    
    for step in range(500):
        state = agent.discretize_state(obs)
        action = agent.act(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Reward shaping
        curr_dist = np.linalg.norm(info['position'] - goal)
        shaped = reward + (prev_dist - curr_dist) * 0.5
        if info.get('is_stuck', False): shaped = -15.0
        prev_dist = curr_dist
        
        agent.learn(state, action, shaped, next_obs)
        obs = next_obs
        if done or truncated: break
    
    agent.epsilon = max(0.05, agent.epsilon * 0.997)


for scenario in ['training_1', 'training_2', 'training_3']:
    test_env = SailingEnv(**get_wind_scenario(scenario))
    print(f"TEST ON SCENARIO {scenario}")
    for episode in range(5):
        observation, info = test_env.reset(seed=22 + episode)
        total_reward      = 0
        reached_goal      = False

        for step in range(1, 501):
            action = agent.act(observation)
            observation, reward, terminated, truncated, info = test_env.step(action)
            total_reward += reward
            dist_to_goal_raw = np.linalg.norm(info['position'] - goal)
            if dist_to_goal_raw < 1.5:
                reached_goal = True

            if terminated or truncated:
                break

        print(f"Test Episode {episode+1}: Steps={step}, Reward={total_reward:.2f}, "
              f"Position={np.round(info['position'], 2)}, Goal reached={reached_goal}")