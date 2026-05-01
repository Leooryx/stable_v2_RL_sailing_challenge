import numpy as np
from src.agents.base_agent import BaseAgent
from src.env_sailing import SailingEnv
from src.wind_scenarios import get_wind_scenario


class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()
        
        self.goal_position = [64, 127]
        self.learning_rate = 0.1
        self.discount_factor = 0.995
        self.exploration_rate = 0.5
        
        self.position_bins = 8
        self.goal_angle_bins = 8
        self.velocity_bins = 4    
        self.wind_angle_bins = 8
        self.danger_bins = 2
        
        self.q_table = {}
        
    def discretize_state(self, observation):
        
        goal_x, goal_y = self.goal_position
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]

        grid_size = 128
        x_bin = min(int(x / grid_size * self.position_bins), self.position_bins - 1)
        y_bin = min(int(y / grid_size * self.position_bins), self.position_bins - 1)
        
        # angle to goal
        dx, dy = goal_x - x, goal_y - y
        angle_to_goal = np.arctan2(dy, dx)
        goal_bin = int(((angle_to_goal + np.pi) / (2 * np.pi) * self.goal_angle_bins)) % self.goal_angle_bins
        
        # relative wind
        wind_angle = np.arctan2(wy, wx)
        rel_wind = (wind_angle - angle_to_goal + np.pi) % (2 * np.pi) - np.pi
        wind_bin = int(((rel_wind + np.pi) / (2 * np.pi) * self.wind_angle_bins)) % self.wind_angle_bins
        
        # discretize velocity direction (ignoring magnitude for simplicity)
        v_magnitude = np.sqrt(vx**2 + vy**2)
        if v_magnitude < 0.1:  # If velocity is very small, consider it as a separate bin
            v_bin = 0
        else:
            v_direction = np.arctan2(vy, vx)  # Range: [-pi, pi]
            v_bin = int(((v_direction + np.pi) / (2 * np.pi) * (self.velocity_bins-1)) + 1) % self.velocity_bins

        # danger radar
        danger_bin = 0
        speed = np.sqrt(vx**2 + vy**2)
        if speed > 0.1:
            look_x = int(np.clip(x + vx * 2, 0, 127))
            look_y = int(np.clip(y + vy * 2, 0, 127))
            map_idx = (6 + 32768) + (look_y * 128 + look_x)
            if map_idx < len(observation) and observation[map_idx] == 1:
                danger_bin = 1
        
        
        return (v_bin, wind_bin, goal_bin, danger_bin)
        
    def act(self, observation):
        """Standardized signature for evaluation."""
        state = self.discretize_state(observation)
        if self.np_random.random() < self.exploration_rate:
            return self.np_random.integers(0, 9)
        if state not in self.q_table:
            return self.default_policy(observation)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        if state not in self.q_table: 
            self.q_table[state] = np.zeros(9)
        if next_state not in self.q_table: 
            self.q_table[next_state] = np.zeros(9)
        
        best_next = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next]
        self.q_table[state][action] += self.learning_rate * (td_target - self.q_table[state][action])

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
    
    def reset(self):
        """Reset the agent"""
        pass

    
    def default_policy(self, obs):
        dx = self.goal_position[0] - obs[0]
        dy = self.goal_position[1] - obs[1]
        angle = np.arctan2(dy, dx)

        # map angle to action
        return int(((angle + np.pi) / (2*np.pi)) * 9) % 9





# training

ql_agent = MyAgent()

np.random.seed(42)
ql_agent.seed(42)
scenarios = ['training_1', 'training_2']
max_steps = 500
num_episodes = 2000

rewards_history = []
steps_history = []
success_history = []

print("Starting training...")

for episode in range(num_episodes):
    scenario = scenarios[episode % 2]
    env = SailingEnv(**get_wind_scenario(scenario))
    goal = env.goal_position.copy()
    
    observation, info = env.reset(seed=episode)
    state = ql_agent.discretize_state(observation)
    
    # Potential function Phi(s) = -distance_to_goal
    prev_dist = np.linalg.norm(info['position'] - goal)
    total_reward = 0
    
    for step in range(max_steps):
        action = ql_agent.act(observation)
        next_observation, base_reward, done, truncated, info = env.step(action)
        next_state = ql_agent.discretize_state(next_observation)
        
        curr_dist = np.linalg.norm(info['position'] - goal)
        
        # potential-based reward shaping
        phi_next = -curr_dist
        phi_curr = -prev_dist
        shaping_reward = (ql_agent.discount_factor * phi_next) - phi_curr
        
        shaped_reward = base_reward + (shaping_reward * 0.2) 
        
        if info.get('is_stuck', False):
            shaped_reward -= 50.0
            
        prev_dist = curr_dist
        
        ql_agent.learn(state, action, shaped_reward, next_state)
        state = next_state
        observation = next_observation
        total_reward += base_reward
        
        if done or truncated:
            break

    rewards_history.append(total_reward)
    steps_history.append(step + 1)
    success_history.append(done)
    
    # Decay exploration exponentially
    ql_agent.exploration_rate = max(0.01, ql_agent.exploration_rate * 0.995)
    
    if (episode + 1) % 200 == 0:
        recent = success_history[-200:]
        print(f"Episode {episode+1}: Success rate (last 200): {sum(recent)/len(recent)*100:.1f}%, Epsilon: {ql_agent.exploration_rate:.3f}")

print(f"Training completed! Q-table size: {len(ql_agent.q_table)} states")
success_rate = sum(success_history) / len(success_history) * 100
print(f"Overall success rate: {success_rate:.1f}%")
print(f"Q-table size: {len(ql_agent.q_table)} states")





# saves the code with the trained weights
from src.utils.agent_utils import save_qlearning_agent
output_path = 'src/my_agent.py'
save_qlearning_agent(ql_agent, output_path, agent_class_name="QLearningTrainedAgent")





#evaluation of performances
ql_agent.exploration_rate = 0

for scenario in ['training_3']:

    # Create test environment
    test_env = SailingEnv(**get_wind_scenario(scenario))

    # added this because goal enters the agent in my implementation
    goal = test_env.goal_position.copy()

    # Test parameters
    num_test_episodes = 5
    max_steps = 500

    print("Testing the trained agent on 5 new episodes...")
    # Testing loop
    for episode in range(num_test_episodes):
        # Reset environment
        observation, info = test_env.reset(seed=1000 + episode)  # Different seeds from training

        
        total_reward = 0
        
        for step in range(max_steps):
            # Select action using learned policy
            action = ql_agent.act(observation)
            observation, reward, done, truncated, info = test_env.step(action)
            
            total_reward += reward
            
            # Break if episode is done
            if done or truncated:
                break
        
        print(f"Test Episode {episode+1}: Steps={step+1}, Reward={total_reward}, " +
            f"Position={info['position']}, Goal reached={done}")