import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from tqdm import tqdm

from src.env_sailing import SailingEnv
from src.wind_scenarios import get_wind_scenario

# =============================================================================
# PyTorch Q-Network
# =============================================================================
class QNetwork(nn.Module):
    def __init__(self, input_dim=10, action_dim=9):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# =============================================================================
# DQN Agent with FNO Look-ahead
# =============================================================================
class MyAgent:
    GRID_SIZE = 128
    
    def __init__(self):
        self.np_random = np.random.default_rng()
        self.goal_position = [64, 127]
        self.exploration_rate = 0.5
        self.discount_factor = 0.995
        
        # Neural Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        
        # DQN Experience Replay Buffers
        self.general_buffer = deque(maxlen=40_000)
        self.fno_buffer = deque(maxlen=10_000)
        self._episode_transitions = []
        self.batch_size = 64
        
        # FNO / Environment caching
        self._world_map = None
        self._wind_field = None
        
        # Initialize FNO if available (assuming same load_fno logic)
        try:
            from src.wind_fno import load_fno
            self._fno = load_fno()
        except ImportError:
            self._fno = None

    # =========================================================================
    # IMPERATIVE INTERFACE (Strict Signatures)
    # =========================================================================
    def act(self, observation):
        if self._world_map is None:
            self._extract_maps(observation)
            
        if self.np_random.random() < self.exploration_rate:
            return int(self.np_random.integers(0, 9))
            
        state = self._extract_continuous_features(observation)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_net(state_t)
            
        return int(torch.argmax(q_values).item())

    def reset(self):
        for t in self._episode_transitions:
            self.general_buffer.append(t)
        self._episode_transitions = []
        self._world_map = None
        self._wind_field = None

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        torch.manual_seed(seed if seed is not None else 42)

    # =========================================================================
    # Continuous State Representation
    # =========================================================================
    def _extract_continuous_features(self, observation):
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]
        gx, gy = self.goal_position

        # Look-ahead path wind
        path_wx, path_wy = 0.0, 0.0
        if self._wind_field is not None:
            mx, my = int(np.clip((x+gx)/2, 0, 127)), int(np.clip((y+gy)/2, 0, 127))
            path_wx, path_wy = self._wind_field[my, mx]

        # Normalize features between ~ -1 and 1
        return np.array([
            x / self.GRID_SIZE, y / self.GRID_SIZE, 
            vx, vy, wx, wy, 
            (gx - x) / self.GRID_SIZE, (gy - y) / self.GRID_SIZE,
            path_wx, path_wy
        ], dtype=np.float32)

    def _extract_maps(self, observation):
        wf_flat = observation[6: 6 + self.GRID_SIZE * self.GRID_SIZE * 2]
        self._wind_field = wf_flat.reshape(self.GRID_SIZE, self.GRID_SIZE, 2).copy()
        ms = 6 + self.GRID_SIZE * self.GRID_SIZE * 2
        self._world_map = observation[ms: ms + self.GRID_SIZE * self.GRID_SIZE].reshape(self.GRID_SIZE, self.GRID_SIZE)

    # =========================================================================
    # DQN Learning & FNO Planning
    # =========================================================================
    def learn(self, observation, action, reward, next_obs, terminal):
        state = self._extract_continuous_features(observation)
        next_state = self._extract_continuous_features(next_obs)
        
        self._episode_transitions.append((state, action, reward, next_state, terminal))
        
        if self._fno is not None:
            self._fno_plan_step(observation)
            
        self._optimize_model()
        
        # Periodically update target network
        if len(self.general_buffer) % 100 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def _optimize_model(self):
        if len(self.general_buffer) < self.batch_size:
            return

        # Mix 70% real experience, 30% FNO synthetic experience
        n_fno = int(self.batch_size * 0.3) if self.fno_buffer else 0
        n_real = self.batch_size - n_fno

        batch = random.sample(self.general_buffer, n_real)
        if n_fno > 0:
            batch += random.sample(self.fno_buffer, n_fno)

        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q-Learning update rule
        q_values = self.q_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states_t).max(1)[0].unsqueeze(1)
            target_q_values = rewards_t + (self.discount_factor * max_next_q_values * (1 - dones_t))

        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _fno_plan_step(self, observation):
        if self._fno is None or self._wind_field is None:
            return
            
        future_fields = self._fno.predict(self._wind_field, steps=1)
        wf_k = future_fields[0]
        
        syn_obs = observation.copy()
        syn_obs[6: 6 + self.GRID_SIZE**2 * 2] = wf_k.flatten()
        ix, iy = int(np.clip(observation[0], 0, 127)), int(np.clip(observation[1], 0, 127))
        syn_obs[4], syn_obs[5] = wf_k[iy, ix, 0], wf_k[iy, ix, 1]

        # Temporarily swap wind field to calculate synthetic continuous state
        old_wf = self._wind_field
        self._wind_field = wf_k
        
        s_k = self._extract_continuous_features(syn_obs)
        
        # Get greedy action from current network for the synthetic state
        with torch.no_grad():
            s_k_t = torch.FloatTensor(s_k).unsqueeze(0).to(self.device)
            action = int(torch.argmax(self.q_net(s_k_t)).item())
            
        self._wind_field = old_wf
        
        syn_reward = 0.05 # Replicate your original synthetic reward shaping
        self.fno_buffer.append((s_k, action, syn_reward, s_k, False))




# =============================================================================
# Helper: Shaped Reward
# =============================================================================
def compute_shaped_reward(base_reward, prev_dist, curr_dist, discount_factor, is_stuck):
    """Potential-based reward shaping to guide the agent toward the goal."""
    shaping = (discount_factor * (-curr_dist)) - (-prev_dist)
    shaped = base_reward + shaping * 0.2
    if is_stuck:
        shaped -= 50.0
    return shaped

# =============================================================================
# Main Train and Test Loop
# =============================================================================
if __name__ == "__main__":
    # 1. Initialize Agent
    agent = MyAgent()
    agent.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 2. Force load FNO from the specific Onyxia path
    FNO_WEIGHTS_PATH = "/home/onyxia/work/stable_v2_RL_sailing_challenge/src/wind_fno_weights.pt"
    try:
        from src.wind_fno import load_fno
        # Assuming load_fno takes a path, or we patch the FNO manually
        agent._fno = load_fno(weights_path=FNO_WEIGHTS_PATH)
        print(f"[FNO] Successfully loaded wind predictor from {FNO_WEIGHTS_PATH}")
    except Exception as e:
        print(f"[FNO Warning] Could not load FNO from specified path. Error: {e}")
        agent._fno = None

    # 3. Training Parameters
    scenarios = ["training_1", "training_2"]
    max_steps = 500
    num_episodes = 2000
    WARMUP_EPISODES = 200
    DECAY_RATE = 0.997
    EPSILON_FLOOR = 0.05

    rewards_history = []
    success_history = []
    prev_scenario = None

    print(f"\nStarting DQN + FNO Training for {num_episodes} episodes...")

    # ---------------------------------------------------------
    # TRAINING LOOP
    # ---------------------------------------------------------
    for episode in tqdm(range(num_episodes)):
        scenario = scenarios[episode % 2]
        
        # Boost exploration slightly when switching environments
        if scenario != prev_scenario and episode > 0:
            agent.exploration_rate = max(agent.exploration_rate, 0.3)
        prev_scenario = scenario

        env = SailingEnv(**get_wind_scenario(scenario))
        goal = env.goal_position.copy()

        observation, info = env.reset(seed=episode)
        agent.reset() # Clears episode buffers and cached maps
        
        prev_dist = np.linalg.norm(info["position"] - goal)
        total_reward = 0

        for step in range(max_steps):
            # Act
            action = agent.act(observation)
            
            # Step environment
            next_obs, base_reward, done, truncated, info = env.step(action)
            terminal = done or truncated
            
            # Calculate metrics for shaping
            curr_dist = np.linalg.norm(info["position"] - goal)
            is_stuck = info.get("is_stuck", False)
            
            shaped_reward = compute_shaped_reward(
                base_reward, prev_dist, curr_dist, 
                agent.discount_factor, is_stuck
            )
            prev_dist = curr_dist

            # Learn (Continuous DQN + FNO Synthetic Replay)
            agent.learn(observation, action, shaped_reward, next_obs, terminal)

            observation = next_obs
            total_reward += base_reward

            if terminal:
                break

        # Post-episode cleanup & tracking
        if not terminal:
            agent.reset()

        rewards_history.append(total_reward)
        success_history.append(done)

        # Decay Epsilon
        if episode >= WARMUP_EPISODES:
            agent.exploration_rate = max(EPSILON_FLOOR, agent.exploration_rate * DECAY_RATE)

        # Logging
        if (episode + 1) % 200 == 0:
            recent_success = sum(success_history[-200:]) / 200.0 * 100
            print(
                f"Episode {episode+1:>4} | "
                f"Success (last 200): {recent_success:5.1f}% | "
                f"ε: {agent.exploration_rate:.3f} | "
                f"Real Buffer: {len(agent.general_buffer):,} | "
                f"FNO Buffer: {len(agent.fno_buffer):,}"
            )

    print("\nTraining completed!")
    print(f"Overall success rate: {sum(success_history)/len(success_history)*100:.1f}%")

    # Save the PyTorch network weights
    save_path = "dqn_fno_agent_weights.pt"
    torch.save(agent.q_net.state_dict(), save_path)
    print(f"Saved DQN weights to {save_path}")

    # ---------------------------------------------------------
    # TESTING LOOP (Held-out scenario)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("Evaluating on held-out test scenario: training_3")
    print("="*60)

    # Disable exploration for testing
    agent.exploration_rate = 0.0
    test_env = SailingEnv(**get_wind_scenario("training_3"))

    for episode in range(5):
        observation, info = test_env.reset(seed=1000 + episode)
        agent.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.act(observation)
            observation, reward, done, truncated, info = test_env.step(action)
            total_reward += reward
            
            if done or truncated:
                break

        print(
            f"Test Episode {episode+1}: "
            f"Steps={step+1:<4} | "
            f"Reward={total_reward:<7.2f} | "
            f"Position={info['position']} | "
            f"Goal Reached={done}"
        )