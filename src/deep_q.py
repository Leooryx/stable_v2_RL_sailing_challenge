import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.agents.base_agent import BaseAgent
from src.env_sailing import SailingEnv
from src.wind_scenarios import get_wind_scenario

# if the ersion of cuda is too old, to this:
"""
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version PyTorch is using: {torch.version.cuda}")

pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
"""

# 1. Define the Neural Network Architecture
class SimpleDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleDQN, self).__init__()
        # A simple 2-layer Multi-Layer Perceptron (MLP)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)


# 2. Define the Deep Q-Learning Agent
class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()
        
        self.goal_position = [64, 127]
        self.learning_rate = 1e-3  
        self.discount_factor = 0.995
        self.exploration_rate = 0.5
        
        # Deep RL specific settings
        self.device = torch.device("cuda")
        self.state_dim = 7  # x, y, vx, vy, wx, wy, danger
        self.action_dim = 9
        self.batch_size = 64
        self.update_target_every = 100 # Steps between target network syncs
        self.steps_done = 0
        
        # Initialize Networks
        self.policy_net = SimpleDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = SimpleDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target net is only used for inference
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Experience Replay Buffer
        self.memory = deque(maxlen=10000)

    def get_state(self, observation):
        """Replaces discretize_state. Returns continuous features as an array."""
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]

        # 1. Goal-Directed Features
        dx, dy = self.goal_position[0] - x, self.goal_position[1] - y
        dx, dy = dx/128, dy/128 #distance normalization
        dist_to_goal = np.sqrt(dx**2 + dy**2)
        angle_to_goal = np.arctan2(dy, dx)

        #relative coordinates are supposed to be better for neural networks
        wind_angle = np.arctan2(wy, wx)
        velocity_angle = np.arctan2(vy, vx)
        # Angle between boat movement and wind
        relative_wind_angle = (velocity_angle - wind_angle + np.pi) % (2 * np.pi) - np.pi


        # danger radar
        danger = 0.0
        speed = np.sqrt(vx**2 + vy**2)
        if speed > 0.1:
            look_x = int(np.clip(x + vx * 2, 0, 127))
            look_y = int(np.clip(y + vy * 2, 0, 127))
            map_idx = (6 + 32768) + (look_y * 128 + look_x)
            if map_idx < len(observation) and observation[map_idx] == 1:
                danger = 1.0
        
        
        return np.array([dist_to_goal, angle_to_goal, vx, vy, relative_wind_angle, wind_angle, danger], dtype=np.float32)
        
    def act(self, observation):
        state = self.get_state(observation)
        
        if self.np_random.random() < self.exploration_rate:
            return int(self.np_random.integers(0, 9))
            
        # Convert state to tensor and ask the network for the best action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            
        return int(torch.argmax(q_values).item())

    def learn(self, state, action, reward, next_state):
        # 1. Store experience in replay memory
        self.memory.append((state, action, reward, next_state))
        
        # 2. Wait until we have enough experiences to train a full batch
        if len(self.memory) < self.batch_size:
            return
            
        # 3. Sample a random batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        # Convert to PyTorch tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        
        # 4. Compute Current Q-Values
        current_q = self.policy_net(states).gather(1, actions)
        
        # 5. Compute Target Q-Values using the Target Network
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.discount_factor * max_next_q)
            
        # 6. Compute loss and optimize
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 7. Periodically sync the target network
        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    
    def reset(self):
        pass

    def default_policy(self, obs):
        dx = self.goal_position[0] - obs[0]
        dy = self.goal_position[1] - obs[1]
        angle = np.arctan2(dy, dx)
        return int(((angle + np.pi) / (2*np.pi)) * 9) % 9


# --- TRAINING LOOP ---

ql_agent = MyAgent()

np.random.seed(42)
ql_agent.seed(42)
scenarios = ['training_1', 'training_2', 'training_3']
max_steps = 500
num_episodes = 2000 #TO BE CHANGED TO 2000

rewards_history = []
steps_history = []
success_history = []

print(f"Starting training on device: {ql_agent.device}")

for episode in tqdm(range(num_episodes)):
    scenario = scenarios[episode % 3]
    env = SailingEnv(**get_wind_scenario(scenario))
    goal = env.goal_position.copy()
    
    observation, info = env.reset(seed=episode)
    
    # Using get_state instead of discretize_state
    state = ql_agent.get_state(observation) 
    
    prev_dist = np.linalg.norm(info['position'] - goal)
    total_reward = 0
    
    
    for step in range(max_steps):
        action = ql_agent.act(observation)
        next_observation, base_reward, done, truncated, info = env.step(action)
        next_state = ql_agent.get_state(next_observation)
        
        # wind-aware reward-shaping
        
        vx, vy = next_observation[2], next_observation[3]
        wx, wy = next_observation[4], next_observation[5]
        wind_angle = np.arctan2(wy, wx)
        vel_angle = np.arctan2(vy, vx) 
        diff = np.abs((vel_angle - wind_angle + np.pi) % (2 * np.pi) - np.pi) 

        # penalty for the no-go zone (45 degrees = 0.78 radians bc arctan)
        no_go_penalty = 0
        if diff < 0.78:
            no_go_penalty = -2.0  # discourages sailing directly into wind      
        
        curr_dist = np.linalg.norm(info['position'] - goal)
        
        phi_next = -curr_dist
        phi_curr = -prev_dist
        shaping_reward = (ql_agent.discount_factor * phi_next) - phi_curr
        
        shaped_reward = base_reward + (shaping_reward * 0.2) + no_go_penalty
        
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
    
    # success-based exploration decay
    recent_success_rate = sum(success_history[-20:]) / 20.0

    if recent_success_rate < 0.1:
        # If we are failing, keep exploration high to find the goal
        ql_agent.exploration_rate = max(0.2, ql_agent.exploration_rate)
    else:
        # If we are succeeding, decay exploration to refine the policy
        ql_agent.exploration_rate = max(0.01, ql_agent.exploration_rate * 0.99)
    
    if (episode + 1) % 200 == 0:
        recent = success_history[-200:]
        print(f"Episode {episode+1}: Success rate (last 200): {sum(recent)/len(recent)*100:.1f}%, Epsilon: {ql_agent.exploration_rate:.3f}")

print("Training completed!")
success_rate = sum(success_history) / len(success_history) * 100
print(f"Overall success rate: {success_rate:.1f}%")


# --- SAVING WEIGHTS ---
output_path = 'src/my_agent_dqn.pth'
torch.save(ql_agent.policy_net.state_dict(), output_path)

import json
def save_weights_as_text(model, filename="weights_dump.txt"):
    # Convert every tensor to a nested Python list
    # .cpu() ensures it's off the GPU, .tolist() converts to raw numbers
    state_dict_raw = {k: v.cpu().tolist() for k, v in model.state_dict().items()}
    
    with open(filename, "w") as f:
        f.write("RAW_WEIGHTS = ")
        # Using json.dumps ensures no truncation and valid Python formatting
        f.write(json.dumps(state_dict_raw))
    
    print(f"Full weights saved to {filename}. Open this file and copy everything!")

# Run this
save_weights_as_text(ql_agent.policy_net)


# --- EVALUATION ---
ql_agent.exploration_rate = 0.0 # Greedy policy for evaluation

for scenario in ['training_3']:
    test_env = SailingEnv(**get_wind_scenario(scenario))
    goal = test_env.goal_position.copy()

    num_test_episodes = 5
    max_steps = 500

    print("Testing the trained agent on 5 new episodes...")
    for episode in range(num_test_episodes):
        observation, info = test_env.reset(seed=22 + episode) 
        total_reward = 0
        
        for step in range(max_steps):
            action = ql_agent.act(observation)
            observation, reward, done, truncated, info = test_env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        print(f"Test Episode {episode+1}: Steps={step+1}, Reward={total_reward:.2f}, " +
              f"Position={np.round(info['position'], 2)}, Goal reached={done}")