import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import json

sys.path.append(os.path.abspath('../src'))
sys.path.append(os.path.abspath('..'))

from agents.base_agent import BaseAgent

np.random.seed(42)




class MLP(nn.Module):
    """Two-hidden-layer MLP using PyTorch."""

    def __init__(self, in_dim, hidden, out_dim, rng=None):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)
        self.tanh = nn.Tanh()
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.fc3.weight, gain=1.0)  
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        # x is a torch tensor
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


# ─────────────────────────────────────────────
#  State featuriser (unchanged)
# ─────────────────────────────────────────────

def featurise(obs):
    """
    Compact, normalised feature vector fed to the networks.
    12 normalized features 

    """

    GRID = 128.0
    x_n = obs[0] / GRID - 0.5
    y_n = obs[1] / GRID - 0.5
    vx, vy = obs[2], obs[3]
    wx, wy = obs[4], obs[5]
    w_norm = np.sqrt(wx**2 + wy**2) + 1e-8
    wx_u, wy_u = wx / w_norm, wy / w_norm
    w_log = np.log1p(w_norm) / 4.0
    gx, gy = 64.0, 127.0
    dx, dy = (gx - obs[0]) / GRID, (gy - obs[1]) / GRID
    dist = np.sqrt(dx**2 + dy**2) + 1e-8
    dx_u, dy_u = dx / dist, dy / dist
    dist_n = dist
    dot_goal_wind = dx_u * wx_u + dy_u * wy_u
    feat = np.array([
        x_n, y_n, vx, vy,
        wx_u, wy_u, w_log,
        dx_u, dy_u, dist_n,
        dot_goal_wind,
    ], dtype=np.float32)
    return feat


FEAT_DIM = 11  


class MyAgent(BaseAgent):
    """
    Minimal PPO with separate actor/critic MLPs, GAE advantage estimation,
    and clipped surrogate loss
    """

    def __init__(
        self,
        hidden=64,
        lr=3e-3,
        gamma=0.995,
        lam=0.95,
        clip_eps=0.2,
        epochs=4,
        ent_coef=0.01,
    ):
        super().__init__()
        self.gamma    = gamma
        self.lam      = lam
        self.clip_eps = clip_eps
        self.epochs   = epochs
        self.ent_coef = ent_coef
        self.lr       = lr

        # PyTorch networks
        self.actor  = MLP(FEAT_DIM, hidden, 9)
        self.critic = MLP(FEAT_DIM, hidden, 1)

        # Use SGD with momentum=0 (matches vanilla gradient descent)
        self.actor_optim  = optim.SGD(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.SGD(self.critic.parameters(), lr=lr)

        # Rollout buffer (filled during one episode, consumed at episode end)
        self._reset_buffer()
        self.np_random = np.random.default_rng()

        torch.manual_seed(42)

    # Obligatoire 

    def reset(self):
        self._reset_buffer()

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        torch.manual_seed(seed)

    def act(self, obs):
        """Inference-only (no gradient). Used at test time."""
        feat = featurise(obs)
        with torch.no_grad():
            feat_t = torch.from_numpy(feat).float()
            logits = self.actor(feat_t)
            probs = torch.softmax(logits, dim=-1).numpy()
        return int(np.argmax(probs))          

    # Training interface

    def act_train(self, obs):
        """Sample action and store transition info for PPO update."""
        feat = featurise(obs)
        with torch.no_grad():
            feat_t = torch.from_numpy(feat).float()
            logits = self.actor(feat_t)
            probs = torch.softmax(logits, dim=-1).numpy()
        action = self.np_random.choice(9, p=probs)
        log_p = np.log(probs[action] + 1e-8)
        with torch.no_grad():
            value = float(self.critic(feat_t).item())
        self._cur = dict(feat=feat, action=action, log_p=log_p, value=value)
        return int(action)

    def store(self, reward, done):
        """Call after env.step() with the shaped reward."""
        self._buf['feats'].append(self._cur['feat'])
        self._buf['actions'].append(self._cur['action'])
        self._buf['log_ps'].append(self._cur['log_p'])
        self._buf['values'].append(self._cur['value'])
        self._buf['rewards'].append(reward)
        self._buf['dones'].append(float(done))

    def finish_episode(self, last_obs):
        """Compute GAE advantages and run PPO epochs. Call at episode end."""
        buf   = self._buf
        T     = len(buf['rewards'])
        if T == 0:
            return

        # Bootstrap value for last step
        last_feat = featurise(last_obs)
        with torch.no_grad():
            last_val = float(self.critic(torch.from_numpy(last_feat).float()).item())

        rewards  = np.array(buf['rewards'],  dtype=np.float32)
        values   = np.array(buf['values'],   dtype=np.float32)
        dones    = np.array(buf['dones'],    dtype=np.float32)

        # GAE (Generalised Advantage Estimation) 
        adv     = np.zeros(T, dtype=np.float32)
        gae     = 0.0
        vals_ext = np.append(values, last_val)
        for t in reversed(range(T)):
            delta  = rewards[t] + self.gamma * vals_ext[t+1] * (1 - dones[t]) - vals_ext[t]
            gae    = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            adv[t] = gae
        returns = adv + values

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        feats   = buf['feats']         
        actions = buf['actions']
        old_lps = np.array(buf['log_ps'], dtype=np.float32)

        # PPO update epochs 
        for _ in range(self.epochs):
            idx = self.np_random.permutation(T)
            for i in idx:
                feat   = feats[i]
                a      = actions[i]
                old_lp = old_lps[i]
                Ai     = adv[i]
                Ri     = returns[i]


                feat_t = torch.from_numpy(feat).float().unsqueeze(0)  # (1, FEAT_DIM)
                old_lp_t = torch.tensor(old_lp, dtype=torch.float32)
                Ai_t = torch.tensor(Ai, dtype=torch.float32)
                Ri_t = torch.tensor(Ri, dtype=torch.float32)

                # Actor loss (clipped surrogate)
                logits = self.actor(feat_t).squeeze(0)   # (9,)
                probs = torch.softmax(logits, dim=-1)
                new_lp = torch.log(probs[a] + 1e-8)
                ratio = torch.exp(new_lp - old_lp_t)
                surr1 = ratio * Ai_t
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * Ai_t
                actor_loss = -torch.min(surr1, surr2)

                # Entropy bonus
                entropy = -torch.sum(probs * torch.log(probs + 1e-8))
                total_loss = actor_loss - self.ent_coef * entropy

                # Update actor
                self.actor_optim.zero_grad()
                total_loss.backward()
                self.actor_optim.step()

                # Critic loss
                v_pred = self.critic(feat_t).squeeze()
                critic_loss = (v_pred - Ri_t) ** 2

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

        self._reset_buffer()


    def save(self, path='ppo_weights.txt'):
        torch_path = path.replace('.txt', '.pt')
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optim_state': self.actor_optim.state_dict(),
            'critic_optim_state': self.critic_optim.state_dict(),
        }, torch_path)

        raw_weights = {
            'actor': {k: v.cpu().tolist() for k, v in self.actor.state_dict().items()},
            'critic': {k: v.cpu().tolist() for k, v in self.critic.state_dict().items()}
        }

        with open(path, "w") as f:
            f.write("RAW_WEIGHTS = ")
            json.dump(raw_weights, f)
        

    def load(self, path='ppo_weights.pt'):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim_state'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim_state'])

    # ── Internal ─────────────────────────────────────────────────────

    def _reset_buffer(self):
        self._buf = {k: [] for k in ('feats', 'actions', 'log_ps', 'values', 'rewards', 'dones')}
        self._cur = {}


### Training:

if __name__ == '__main__':
    from tqdm import tqdm
    from env_sailing import SailingEnv
    from wind_scenarios import get_wind_scenario

    SCENARIOS_TRAIN   = ['training_1', 'training_2']
    SCENARIO_VALID = 'training_3'

    NUM_EPISODES = 600
    GOAL         = np.array([64.0, 127.0])

    agent = MyAgent(hidden=64, lr=3e-3, gamma=0.995, lam=0.95,
                    clip_eps=0.2, epochs=4, ent_coef=0.01)
    agent.seed(42)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(agent.actor_optim, mode='max', factor=0.5, patience=20, verbose=True)
    # we also need to apply it to critic!

    best_success_rate = -1.0
    best_avg_steps = float('inf')
    best_episode = -1

    success_log = []
    steps_log = []

    def validate(agent, num_episodes=10, max_steps=500):
        successes = 0
        steps_list = []
        for ep in range(num_episodes):
            env = SailingEnv(**get_wind_scenario(SCENARIO_VALID))
            obs, info = env.reset(seed=ep + 1000)
            reached = False
            for step in range(1, max_steps +1):
                action = agent.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                if np.linalg.norm(info['position'] - GOAL) < 1.5:
                    reached = True
                    break
                if terminated or truncated:
                    break
            if reached:
                successes += 1
                steps_list.append(step)
        success_rate = successes / num_episodes
        avg_steps = np.mean(steps_list) if steps_list else float('inf')
        return success_rate, avg_steps


    for episode in tqdm(range(NUM_EPISODES), ):
        scenario = SCENARIOS_TRAIN[episode % len(SCENARIOS_TRAIN)]
        env = SailingEnv(**get_wind_scenario(scenario))
        obs, info = env.reset(seed=episode)
        agent.reset()

        prev_dist = np.linalg.norm(info['position'] - GOAL)
        done_flag = False
        steps_taken = 0

        for step in range(500):
            action = agent.act_train(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps_taken = step + 1

            curr_dist = np.linalg.norm(info['position'] - GOAL)
            shaped    = reward + (prev_dist - curr_dist) * 0.5
            if info.get('is_stuck', False):
                shaped = -15.0
            prev_dist = curr_dist

            agent.store(shaped, done)
            obs = next_obs

            if done:
                done_flag = terminated   # True only if goal reached
                break

        agent.finish_episode(obs)
        success_log.append(done_flag)

        if done_flag:
            steps_log.append(steps_taken)
        else:
            steps_log.append(None)

        if (episode + 1) % 100 == 0:
            rate = sum(success_log[-100:]) / 100
            print(f"Episode {episode+1:4d} | success rate (last 100): {rate:.0%}")

        #validation
        if (episode +1) % 20 == 0:
            val_success, val_avg_steps = validate(agent, num_episodes=10)
            print(f"Validation at episode {episode+1}: sucess={val_success:.2f}, avg_steps={val_avg_steps:.1f}")
            scheduler.step(val_success)

            is_better = False
            if val_success > best_success_rate:
                is_better = True
                print("val_success", val_success)
                print("best_success_rate", best_success_rate)
            elif val_success == best_success_rate and val_avg_steps < best_avg_steps: 
                # sur que égalité pose pas de pb?
                is_better = True
                print("better found!")
            
            if is_better:
                best_success_rate = val_success
                best_avg_steps = val_avg_steps
                best_episode = episode
                agent.save()
    

    # normally evaluation should use the last loaded model, so the best right?

    ### Evaluation 

    agent.load() #loads the best
    for scenario in ['training_1', 'training_2', 'training_3']:
        test_env = SailingEnv(**get_wind_scenario(scenario))
        print(f"TEST ON SCENARIO {scenario}")
        for episode in range(5):
            observation, info = test_env.reset(seed=22 + episode)
            total_reward = 0
            reached_goal = False

            for step in range(1, 501):
                action = agent.act(observation)
                observation, reward, terminated, truncated, info = test_env.step(action)
                total_reward += reward
                if np.linalg.norm(info['position'] - GOAL) < 1.5:
                    reached_goal = True
                if terminated or truncated:
                    break

            disc = total_reward * (0.995 ** (step - 1)) if reached_goal else 0 # i should print this instead
            print(f"  Ep {episode+1}: steps={step:3d}  reward={total_reward:.1f}"
                  f"  pos={np.round(info['position'],1)}  reached={reached_goal}")
        print()