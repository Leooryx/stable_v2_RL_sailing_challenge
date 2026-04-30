import numpy as np
from src.agents.base_agent import BaseAgent
from src.env_sailing import SailingEnv
from src.wind_scenarios import get_wind_scenario


class ImprovedQLearningAgent(BaseAgent):
    """A smarter Q-learning agent using relative state abstractions."""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.995, exploration_rate=0.5):
        self.np_random = np.random.default_rng()
        
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # We drastically reduce the state space complexity:
        self.goal_angle_bins = 8   # 8 directions to the goal
        self.wind_angle_bins = 8   # 8 relative wind directions
        self.danger_bins = 2       # 0 = safe, 1 = island directly ahead
        
        self.q_table = {}
        
    def discretize_state(self, observation, goal_position):
        """Convert continuous observation to a smart, relative discrete state."""
        goal_x, goal_y = goal_position
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]
        
        # 1. Angle to goal (Where do I want to go?)
        dx, dy = goal_x - x, goal_y - y
        angle_to_goal = np.arctan2(dy, dx)
        goal_bin = int(((angle_to_goal + np.pi) / (2 * np.pi) * self.goal_angle_bins)) % self.goal_angle_bins
        
        # 2. Wind direction relative to the goal (Do I have a headwind?)
        # This tells the agent if it's facing the No-Go Zone for its intended path
        wind_angle = np.arctan2(wy, wx)
        rel_wind = wind_angle - angle_to_goal
        rel_wind = (rel_wind + np.pi) % (2 * np.pi) - np.pi # Normalize to [-pi, pi]
        wind_bin = int(((rel_wind + np.pi) / (2 * np.pi) * self.wind_angle_bins)) % self.wind_angle_bins
        
        # 3. Danger Radar (Am I about to crash?)
        # We look 2 steps ahead in our current direction of velocity
        danger_bin = 0
        speed = np.sqrt(vx**2 + vy**2)
        if speed > 0.1:
            look_x = int(np.clip(x + vx * 2, 0, 127))
            look_y = int(np.clip(y + vy * 2, 0, 127))
            
            # Extract map data from observation (Indices: 0-5 core, 6 to 6+32767 wind, 6+32768+ map)
            map_start_idx = 6 + 32768
            map_idx = map_start_idx + (look_y * 128 + look_x)
            
            # Check if out of bounds or if it's an island
            if map_idx < len(observation):
                if observation[map_idx] == 1:
                    danger_bin = 1
        
        return (goal_bin, wind_bin, danger_bin)
        
    def act(self, observation, goal):
        state = self.discretize_state(observation, goal)
        
        if self.np_random.random() < self.exploration_rate:
            return self.np_random.integers(0, 9)
        else:
            if state not in self.q_table:
                self.q_table[state] = np.zeros(9)
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(9)
            
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
        
    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)





# training

ql_agent = ImprovedQLearningAgent(
    learning_rate=0.1, 
    discount_factor=0.995, 
    exploration_rate=0.5 # Start with 100% exploration
)

np.random.seed(42)
ql_agent.seed(42)
scenarios = ['training_1', 'training_2', 'training_3']
max_steps = 500
num_episodes = 2000

rewards_history = []
steps_history = []
success_history = []

print("Starting training...")

for episode in range(num_episodes):
    scenario = scenarios[episode % 3]
    env = SailingEnv(**get_wind_scenario(scenario))
    goal = env.goal_position.copy()
    
    observation, info = env.reset(seed=episode)
    state = ql_agent.discretize_state(observation, goal)
    
    # Potential function Phi(s) = -distance_to_goal
    prev_dist = np.linalg.norm(info['position'] - goal)
    total_reward = 0
    
    for step in range(max_steps):
        action = ql_agent.act(observation, goal)
        next_observation, base_reward, done, truncated, info = env.step(action)
        next_state = ql_agent.discretize_state(next_observation, goal)
        
        curr_dist = np.linalg.norm(info['position'] - goal)
        
        # --- POTENTIAL-BASED REWARD SHAPING ---
        # Formula: F(s, s') = gamma * Phi(s') - Phi(s)
        phi_next = -curr_dist
        phi_curr = -prev_dist
        shaping_reward = (ql_agent.discount_factor * phi_next) - phi_curr
        
        # Scale the shaping to balance with the final +100 goal reward
        shaped_reward = base_reward + (shaping_reward * 0.2) 
        
        # Heavy penalty for hard failures (islands)
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


from src.utils.agent_utils import save_qlearning_agent
output_path = 'src/saved_agent.py'
save_qlearning_agent(ql_agent, output_path, agent_class_name="QLearningTrainedAgent")