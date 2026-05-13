import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.abspath('../src'))
sys.path.append(os.path.abspath('..'))

from agents.base_agent import BaseAgent
from env_sailing import SailingEnv
from wind_scenarios import get_wind_scenario



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



goal = [64, 127]


# ── 1. Actor-Critic Network ───────────────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor  = nn.Linear(64, output_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        features = self.backbone(x)
        return self.actor(features), self.critic(features)

    def get_action(self, x):
        logits, value = self.forward(x)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


# ── 2. Rollout Buffer ─────────────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states, self.actions, self.rewards = [], [], []
        self.log_probs, self.values, self.terminatedes  = [], [], []

    def add(self, state, action, reward, log_prob, value, terminated):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.terminatedes.append(terminated)

    def compute_returns(self, gamma, gae_lambda, last_value, device):
        T          = len(self.rewards)
        advantages = torch.zeros(T, device=device)
        last_gae   = 0.0

        for t in reversed(range(T)):
            next_val   = last_value if t == T - 1 else self.values[t + 1]
            not_terminated   = 1.0 - float(self.terminatedes[t])
            delta      = self.rewards[t] + gamma * next_val * not_terminated - self.values[t]
            last_gae   = delta + gamma * gae_lambda * not_terminated * last_gae
            advantages[t] = last_gae

        returns = advantages + torch.stack(self.values)
        return advantages, returns


# ── 3. PPO Agent ──────────────────────────────────────────────────────────────
class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.np_random      = np.random.default_rng()
        self.goal_position  = [64, 127]

        self.gamma          = 0.995
        self.gae_lambda     = 0.95
        self.clip_eps       = 0.2
        self.lr             = 3e-4
        self.n_epochs       = 4
        self.batch_size     = 64
        self.entropy_coef   = 0.01
        self.value_coef     = 0.5

        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim  = 7
        self.action_dim = 9

        self.ac        = ActorCritic(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.lr)

        self.buffer      = RolloutBuffer()
        self.steps_done  = 0

    def get_state(self, observation):
        x, y   = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]

        dx, dy        = (self.goal_position[0] - x) / 128, (self.goal_position[1] - y) / 128
        dist_to_goal  = np.sqrt(dx**2 + dy**2)
        angle_to_goal = np.arctan2(dy, dx)

        wind_angle          = np.arctan2(wy, wx)
        velocity_angle      = np.arctan2(vy, vx)
        relative_wind_angle = (velocity_angle - wind_angle + np.pi) % (2 * np.pi) - np.pi

        danger = 0.0
        speed  = np.sqrt(vx**2 + vy**2)
        if speed > 0.1:
            look_x  = int(np.clip(x + vx * 2, 0, 127))
            look_y  = int(np.clip(y + vy * 2, 0, 127))
            map_idx = (6 + 32768) + (look_y * 128 + look_x)
            if map_idx < len(observation) and observation[map_idx] == 1:
                danger = 1.0

        return np.array(
            [dist_to_goal, angle_to_goal, vx, vy, relative_wind_angle, wind_angle, danger],
            dtype=np.float32
        )

    def act(self, observation):
        state  = self.get_state(observation)
        tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, _, value = self.ac.get_action(tensor)
        self._last_state    = state
        self._last_log_prob = log_prob.item()
        self._last_value    = value.squeeze().item()
        return int(action.item())

    def learn(self, state, action, reward, next_state, terminated=False):
        self.buffer.add(
            state    = torch.FloatTensor(state).to(self.device),
            action   = torch.tensor(action,              device=self.device),
            reward   = reward,
            log_prob = torch.tensor(self._last_log_prob, device=self.device),
            value    = torch.tensor(self._last_value,    device=self.device),
            terminated     = terminated,
        )
        self.steps_done += 1

        if done:
            with torch.no_grad():
                ns_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(ppo_agent.device)
                _, last_v = ppo_agent.ac(ns_tensor)
                last_value = last_v.squeeze().item() if truncated or not terminated else 0.0
            ppo_agent._update(last_value)
            ppo_agent.buffer.clear()

    def _update(self, last_value):
        advantages, returns = self.buffer.compute_returns(
            self.gamma, self.gae_lambda, last_value, self.device
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states  = torch.stack(self.buffer.states)
        actions = torch.stack(self.buffer.actions)
        old_lps = torch.stack(self.buffer.log_probs).detach()

        T = len(self.buffer.states)
        for _ in range(self.n_epochs):
            indices = torch.randperm(T)
            for start in range(0, T, self.batch_size):
                idx = indices[start : start + self.batch_size]

                logits, values_pred = self.ac(states[idx])
                dist    = Categorical(logits=logits)
                new_lps = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()

                ratio      = (new_lps - old_lps[idx]).exp()
                surr1      = ratio * advantages[idx]
                surr2      = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages[idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(values_pred.squeeze(), returns[idx])

                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.optimizer.step()

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def reset(self):
        pass

    def default_policy(self, obs):
        dx    = self.goal_position[0] - obs[0]
        dy    = self.goal_position[1] - obs[1]
        angle = np.arctan2(dy, dx)
        return int(((angle + np.pi) / (2 * np.pi)) * 9) % 9


# ── 4. Training Loop ──────────────────────────────────────────────────────────
ppo_agent = MyAgent()

np.random.seed(42)
ppo_agent.seed(42)

scenarios    = ['training_1', 'training_2']
num_episodes = 600

rewards_history = []
steps_history   = []
success_history = []

print(f"Starting PPO training on device: {ppo_agent.device}")

for episode in tqdm(range(num_episodes)):
    scenario          = scenarios[episode % 2]
    env               = SailingEnv(**get_wind_scenario(scenario))
    goal              = env.goal_position.copy()
    observation, info = env.reset(seed=episode)
    state             = ppo_agent.get_state(observation)
    prev_dist         = np.linalg.norm(info['position'] - goal)
    total_reward      = 0

    for step in range(1, 501):
        action = ppo_agent.act(observation)
        next_observation, base_reward, terminated, truncated, info = env.step(action)
        next_state = ppo_agent.get_state(next_observation)

        #if terminated: discounted = 100 * (discount_factor ** (step - 1))

        # ── Goal detection ────────────────────────────────────────────────────
        # get_state computes dist_to_goal on the normalised [0,1] scale.
        # The raw distance threshold of 1.5 on a 128-unit grid corresponds to
        # 1.5/128 ≈ 0.01172 in normalised units.
        dist_to_goal_raw = np.linalg.norm(info['position'] - goal)
        reached_goal     = dist_to_goal_raw < 1.5

        # Alternatively: terminated=True AND step < 501 means goal was reached
        # (not stuck, because stuck only happens after 500 steps → step == 500
        # with terminated=True; goal can be reached at any step 1..500).
        # Both checks are equivalent; we keep the explicit distance check as the
        # canonical signal and use `terminated and step < 500` as a safeguard.

        # `done` for the GAE bootstrap: True whenever the episode truly ends
        # (goal reached, stuck, or crashed into island).
        # terminated = goal reached OR stuck (env signal)
        # truncated  = hit the 500-step wall (episode cut short, NOT a terminal state)
        # A crash into an island also sets terminated=True in the env.
        done = terminated  # do NOT include truncated — truncated is NOT a terminal state

        # ── Reward shaping ────────────────────────────────────────────────────
        vx, vy     = next_observation[2], next_observation[3]
        wx, wy     = next_observation[4], next_observation[5]
        speed      = np.sqrt(vx**2 + vy**2)
        wind_angle = np.arctan2(wy, wx)
        vel_angle  = np.arctan2(vy, vx)
        diff       = np.abs((vel_angle - wind_angle + np.pi) % (2 * np.pi) - np.pi)
        no_go_penalty = -0.5 if (diff < 0.78 and speed > 0.05) else 0.0

        curr_dist      = np.linalg.norm(info['position'] - goal)
        phi_next       = -curr_dist
        phi_curr       = -prev_dist
        shaping_reward = (ppo_agent.gamma * phi_next) - phi_curr
        shaped_reward  = base_reward + (shaping_reward * 0.3) + no_go_penalty

        if info.get('is_stuck', False):
            shaped_reward -= 5.0

        prev_dist = curr_dist

        # Pass `done` (= terminated) so GAE correctly zeroes the bootstrap value
        # at true terminal states. When truncated (500-step cutoff), done=False
        # so the critic's value estimate is used to bootstrap — correct behaviour.
        ppo_agent.learn(state, action, shaped_reward, next_state, terminated=terminated)

        state       = next_state
        observation = next_observation
        total_reward += base_reward

        if terminated:
            break

    rewards_history.append(total_reward)
    steps_history.append(step)
    success_history.append(reached_goal)

    if (episode + 1) % 100 == 0:
        recent = success_history[-100:]
        print(f"Episode {episode+1}: Success rate (last 200): {sum(recent)/len(recent)*100:.1f}%")

print("Training completed!")
print(f"Overall success rate: {sum(success_history)/len(success_history)*100:.1f}%")


# ── 5. Save weights ───────────────────────────────────────────────────────────
torch.save(ppo_agent.ac.state_dict(), 'my_agent_ppo.pth')

import json
def save_weights_as_text(model, filename="weights_dump_ppo.txt"):
    state_dict_raw = {k: v.cpu().tolist() for k, v in model.state_dict().items()}
    with open(filename, "w") as f:
        f.write("RAW_WEIGHTS = ")
        f.write(json.dumps(state_dict_raw))
    print(f"Full weights saved to {filename}.")

save_weights_as_text(ppo_agent.ac)


# ── 6. Evaluation ─────────────────────────────────────────────────────────────
for scenario in ['training_1', 'training_2', 'training_3']:
    test_env = SailingEnv(**get_wind_scenario(scenario))

    print(f"\nTesting the trained PPO agent on 5 episodes for scenario {scenario}...")
    for episode in range(5):
        observation, info = test_env.reset(seed=22 + episode)
        total_reward      = 0
        reached_goal      = False

        for step in range(1, 501):
            action = ppo_agent.act(observation)
            observation, reward, terminated, truncated, info = test_env.step(action)
            total_reward += reward
            dist_to_goal_raw = np.linalg.norm(info['position'] - goal)
            if dist_to_goal_raw < 1.5 or (terminated and step < 500):
                reached_goal = True

            if terminated or truncated:
                break

        print(f"Test Episode {episode+1}: Steps={step}, Reward={total_reward:.2f}, "
              f"Position={np.round(info['position'], 2)}, Goal reached={terminated}")