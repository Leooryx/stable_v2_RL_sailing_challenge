"""
my_agent.py  —  Dyna-Q agent with FNO wind look-ahead planning
================================================================
Adds an FNO-powered planning module on top of the fixed Dyna-Q
agent (v2). The FNO predicts the wind field 1–5 steps into the
future; those synthetic transitions feed the planning buffer and
let the Q-table learn from wind conditions it hasn't yet visited.

How the FNO integrates into Dyna-Q
------------------------------------
Standard Dyna-Q planning loop:
    sample (s, a, r, s') from experience buffer
    → Q-update

FNO-augmented planning loop (new):
    At each real step:
      1. Predict wind fields at t+1 … t+5 using FNO
      2. For each predicted future wind w_k:
         - Build a synthetic observation by swapping the current
           wind field for the predicted one (position/velocity stay)
         - Discretize → synthetic state s_k
         - Compute a synthetic reward via the shaped reward function
         - Store (s_k, best_action, r_synthetic, s_{k+1}) in a
           separate fno_buffer
      3. During planning, sample 30% from fno_buffer (wind-aware
         future states) and 70% from the regular split buffer
         (real experience)

This gives the agent Q-values for states it will encounter in the
NEAR FUTURE, not just states it has visited before. Critical for
training_3 generalization: even though training_3's initial wind
field is new, the FNO predicts its evolution accurately (same
dynamics, different initial condition).

Fallback behaviour
------------------
If FNO weights don't exist yet (first training run), the agent
falls back to the standard Dyna-Q planning loop silently.
Set USE_FNO = False to disable permanently.
"""

import numpy as np
from collections import deque
from pathlib import Path
from tqdm import tqdm
from src.agents.base_agent import BaseAgent
from src.env_sailing import SailingEnv
from src.wind_scenarios import get_wind_scenario

USE_FNO = True   # set False to run without FNO (for ablation)
FNO_STEPS = 1    # how many steps ahead to predict
FNO_PLAN_FRACTION = 0.30  # fraction of planning budget from FNO buffer

try:
    if USE_FNO:
        from src.wind_fno import load_fno, WEIGHTS_PATH
        _fno_available = WEIGHTS_PATH.exists()
    else:
        _fno_available = False
except ImportError:
    _fno_available = False


# =============================================================================
# Agent
# =============================================================================

class MyAgent(BaseAgent):

    GRID_SIZE = 128
    GOAL = np.array([64, 127])

    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()

        self.goal_position = [64, 127]
        self.learning_rate = 0.15
        self.discount_factor = 0.995
        self.exploration_rate = 0.5

        # State discretisation
        self.position_bins    = 16
        self.velocity_dir_bins = 8
        self.wind_angle_bins  = 8
        self.path_wind_bins   = 4

        # Q-table
        self.q_table = {}

        # Split experience buffer (v2 fix)
        self.general_buffer  = deque(maxlen=40_000)
        self.success_buffer  = deque(maxlen=20_000)
        self._episode_transitions = []

        # FNO synthetic buffer — populated each step with predicted transitions
        self.fno_buffer = deque(maxlen=10_000)

        self.planning_steps   = 50
        self.priority_fraction = 0.20

        # Cached maps
        self._world_map  = None
        self._wind_field = None   # (128, 128, 2) numpy

        # FNO inference wrapper (None if not available)
        self._fno: "WindFNOInference | None" = None
        if _fno_available:
            try:
                from src.wind_fno import load_fno
                self._fno = load_fno()
                print("[FNO] Loaded wind predictor.")
            except Exception as e:
                print(f"[FNO] Could not load: {e}. Falling back to standard planning.")

    # =========================================================================
    # Required BaseAgent interface
    # =========================================================================

    def act(self, observation):
        if self._world_map is None:
            self._extract_maps(observation)
        state = self.discretize_state(observation)

        if self.np_random.random() < self.exploration_rate:
            return int(self.np_random.integers(0, 9))
        if state not in self.q_table:
            return self.default_policy(observation)
        return int(np.argmax(self.q_table[state]))

    def reset(self):
        for t in self._episode_transitions:
            self.general_buffer.append(t)
        self._episode_transitions = []
        self._world_map  = None
        self._wind_field = None

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)

    # =========================================================================
    # State discretisation
    # =========================================================================

    def discretize_state(self, observation):
        x,  y  = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]
        gx, gy = self.goal_position

        x_bin = int(np.clip(x / self.GRID_SIZE * self.position_bins, 0, self.position_bins - 1))
        y_bin = int(np.clip(y / self.GRID_SIZE * self.position_bins, 0, self.position_bins - 1))

        v_mag = np.sqrt(vx**2 + vy**2)
        if v_mag < 0.15:
            vel_bin = 0
        else:
            dir_bin = int(((np.arctan2(vy, vx) + np.pi) / (2 * np.pi) * self.velocity_dir_bins)) % self.velocity_dir_bins
            mag_tier = 1 if v_mag < 0.8 else 2
            vel_bin = mag_tier * self.velocity_dir_bins + dir_bin

        atg = np.arctan2(gy - y, gx - x)
        rel_wind = (np.arctan2(wy, wx) - atg + np.pi) % (2 * np.pi) - np.pi
        local_wind_bin = int(((rel_wind + np.pi) / (2 * np.pi) * self.wind_angle_bins)) % self.wind_angle_bins

        path_wind_bin = self._path_wind_bin(x, y, gx, gy)

        danger_bin = 0
        if v_mag > 0.1 and self._world_map is not None:
            lx = int(np.clip(x + vx * 2, 0, self.GRID_SIZE - 1))
            ly = int(np.clip(y + vy * 2, 0, self.GRID_SIZE - 1))
            if self._world_map[ly, lx] == 1:
                danger_bin = 1

        return (x_bin, y_bin, vel_bin, local_wind_bin, path_wind_bin, danger_bin)

    def _path_wind_bin(self, x, y, gx, gy):
        if self._wind_field is None:
            return 0
        awx = awy = 0.0
        for t in [0.25, 0.5, 0.75, 1.0]:
            px = int(np.clip(x + t * (gx - x), 0, self.GRID_SIZE - 1))
            py = int(np.clip(y + t * (gy - y), 0, self.GRID_SIZE - 1))
            awx += self._wind_field[py, px, 0]
            awy += self._wind_field[py, px, 1]
        return int(((np.arctan2(awy, awx) + np.pi) / (2 * np.pi) * self.path_wind_bins)) % self.path_wind_bins

    # =========================================================================
    # FNO planning: generate synthetic future transitions
    # =========================================================================

    def _fno_plan_step(self, observation):
        """
        Use the FNO to predict FNO_STEPS future wind fields, build synthetic
        observations for each, discretize them, and push to fno_buffer.

        Called once per real environment step (inside learn()).

        Synthetic transition structure:
            For each predicted step k in 1..FNO_STEPS:
              - s_k   = discretize(obs with wind_field replaced by predicted_k-1)
              - s_k1  = discretize(obs with wind_field replaced by predicted_k)
              - action = best greedy action from s_k (no exploration)
              - reward = shaped reward based on position shaping only
                         (we don't know the real outcome, so we use Φ-shaping)
        """
        if self._fno is None or self._wind_field is None:
            return

        # Predict future wind fields: shape (FNO_STEPS, 128, 128, 2)
        future_fields = self._fno.predict(self._wind_field, steps=FNO_STEPS)

        x, y = observation[0], observation[1]
        goal = np.array(self.goal_position, dtype=float)

        # Build a synthetic observation template (we only swap the wind field)
        # The position and velocity stay fixed — the FNO only tells us about wind
        syn_obs = observation.copy()

        prev_dist = np.linalg.norm(np.array([x, y]) - goal)

        for k in range(FNO_STEPS):
            # Wind at step k (index into future_fields)
            wf_k = future_fields[k]                  # (128, 128, 2)

            # Swap wind field into synthetic observation
            wf_flat = wf_k.flatten()
            syn_obs[6: 6 + self.GRID_SIZE * self.GRID_SIZE * 2] = wf_flat

            # Local wind at current position (y then x indexing)
            ix = int(np.clip(x, 0, self.GRID_SIZE - 1))
            iy = int(np.clip(y, 0, self.GRID_SIZE - 1))
            syn_obs[4] = wf_k[iy, ix, 0]
            syn_obs[5] = wf_k[iy, ix, 1]

            # Also update cached wind field so _path_wind_bin uses future wind
            old_wf = self._wind_field
            self._wind_field = wf_k

            s_k = self.discretize_state(syn_obs)

            # For s_{k+1}, use the next predicted field (or same if at end)
            if k + 1 < FNO_STEPS:
                wf_k1 = future_fields[k + 1]
                syn_obs_next = syn_obs.copy()
                syn_obs_next[6: 6 + self.GRID_SIZE * self.GRID_SIZE * 2] = wf_k1.flatten()
                ix1 = int(np.clip(x, 0, self.GRID_SIZE - 1))
                iy1 = int(np.clip(y, 0, self.GRID_SIZE - 1))
                syn_obs_next[4] = wf_k1[iy1, ix1, 0]
                syn_obs_next[5] = wf_k1[iy1, ix1, 1]
                self._wind_field = wf_k1
                s_k1 = self.discretize_state(syn_obs_next)
            else:
                s_k1 = s_k   # terminal of the FNO rollout

            # Restore real wind field
            self._wind_field = old_wf

            # Best action from current Q-table (greedy, no exploration)
            if s_k in self.q_table:
                action = int(np.argmax(self.q_table[s_k]))
            else:
                action = self.default_policy(syn_obs)

            # Reward: purely potential-based (we don't know the real outcome)
            # Using the same shaping formula: F = γΦ(s') - Φ(s)
            # Position doesn't change (wind prediction only), so shaping = 0
            # unless we add wind-aware component. For now: small constant reward
            # proportional to wind alignment with goal direction.
            wx_k = wf_k[iy, ix, 0]
            wy_k = wf_k[iy, ix, 1]
            dx = goal[0] - x
            dy = goal[1] - y
            dist = np.sqrt(dx**2 + dy**2) + 1e-8
            # Wind alignment with goal: cos(angle between wind and goal direction)
            wind_goal_align = (wx_k * dx + wy_k * dy) / (dist * (np.sqrt(wx_k**2 + wy_k**2) + 1e-8))
            syn_reward = 0.05 * wind_goal_align   # small synthetic signal

            self.fno_buffer.append((s_k, action, syn_reward, s_k1, False))

    # =========================================================================
    # Learning: Q-update + Dyna-Q planning
    # =========================================================================

    def learn(self, state, action, shaped_reward, next_state, terminal=False, observation=None):
        """
        Real Q-update + planning (with optional FNO synthetic transitions).

        observation: pass the current raw observation to enable FNO planning.
                     If None, FNO step is skipped.
        """
        # Real Q-update
        self._q_update(state, action, shaped_reward, next_state, terminal)

        # Accumulate episode transitions
        self._episode_transitions.append(
            (state, action, shaped_reward, next_state, terminal)
        )

        # Route completed episode to correct buffer
        if terminal:
            goal_reached = (shaped_reward > 50)
            target = self.success_buffer if goal_reached else self.general_buffer
            for t in self._episode_transitions:
                target.append(t)
            self._episode_transitions = []

        # FNO: generate synthetic future transitions
        if observation is not None and self._fno is not None:
            self._fno_plan_step(observation)

        # Planning budget
        total = len(self.general_buffer) + len(self.success_buffer)
        if total < 10:
            return

        # Split planning budget: FNO / priority / uniform
        if self._fno is not None and len(self.fno_buffer) > 0:
            n_fno      = int(self.planning_steps * FNO_PLAN_FRACTION)
            remaining  = self.planning_steps - n_fno
            n_priority = int(remaining * self.priority_fraction)
            n_uniform  = remaining - n_priority
        else:
            n_fno      = 0
            n_priority = int(self.planning_steps * self.priority_fraction)
            n_uniform  = self.planning_steps - n_priority

        self._plan_fno(n_fno)
        self._plan_uniform(n_uniform)
        self._plan_priority(n_priority)

    def _plan_fno(self, n):
        """Sample from FNO synthetic buffer."""
        if n == 0 or not self.fno_buffer:
            return
        for _ in range(n):
            idx = int(self.np_random.integers(0, len(self.fno_buffer)))
            s, a, r, sn, term = self.fno_buffer[idx]
            self._q_update(s, a, r, sn, term)

    def _plan_uniform(self, n):
        buffers = [b for b in [self.general_buffer, self.success_buffer] if b]
        if not buffers:
            return
        for i in range(n):
            buf = buffers[i % len(buffers)]
            idx = int(self.np_random.integers(0, len(buf)))
            s, a, r, sn, t = buf[idx]
            self._q_update(s, a, r, sn, t)

    def _plan_priority(self, n):
        if not self.success_buffer or n == 0:
            return
        high_r = [(s, a, r, sn, t) for s, a, r, sn, t in self.success_buffer if abs(r) > 10]
        src = high_r if high_r else list(self.success_buffer)
        for idx in self.np_random.integers(0, len(src), size=n):
            s, a, r, sn, t = src[idx]
            self._q_update(s, a, r, sn, t)

    def _q_update(self, state, action, reward, next_state, terminal):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(9)
        tgt = (reward if terminal
               else reward + self.discount_factor * np.max(self.q_table[next_state]))
        self.q_table[state][action] += self.learning_rate * (tgt - self.q_table[state][action])

    # =========================================================================
    # Default / fallback policy
    # =========================================================================

    def default_policy(self, obs):
        dx = self.goal_position[0] - obs[0]
        dy = self.goal_position[1] - obs[1]
        return int(((np.arctan2(dy, dx) + np.pi) / (2 * np.pi)) * 9) % 9

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _extract_maps(self, observation):
        wf_flat = observation[6: 6 + self.GRID_SIZE * self.GRID_SIZE * 2]
        self._wind_field = wf_flat.reshape(self.GRID_SIZE, self.GRID_SIZE, 2).copy()
        ms = 6 + self.GRID_SIZE * self.GRID_SIZE * 2
        self._world_map = observation[ms: ms + self.GRID_SIZE * self.GRID_SIZE].reshape(
            self.GRID_SIZE, self.GRID_SIZE
        )


# =============================================================================
# Training pipeline
# =============================================================================

def compute_shaped_reward(base_reward, prev_dist, curr_dist, discount_factor, is_stuck):
    shaping = (discount_factor * (-curr_dist)) - (-prev_dist)
    shaped = base_reward + shaping * 0.2
    if is_stuck:
        shaped -= 50.0
    return shaped


if __name__ == "__main__":

    agent = MyAgent()
    np.random.seed(42)
    agent.seed(42)

    scenarios   = ["training_1", "training_2"]
    max_steps   = 500
    num_episodes = 2000

    rewards_history = []
    steps_history   = []
    success_history = []

    WARMUP_EPISODES = 200
    DECAY_RATE      = 0.997
    EPSILON_FLOOR   = 0.05

    prev_scenario = None

    fno_label = "FNO ON" if agent._fno is not None else "FNO OFF (run wind_fno.py first)"
    print(f"Starting Dyna-Q + FNO training  [{fno_label}]")
    print(f"  Planning steps : {agent.planning_steps}  "
          f"(FNO={int(agent.planning_steps * FNO_PLAN_FRACTION)} | "
          f"priority={int(agent.planning_steps * agent.priority_fraction)} | "
          f"uniform=rest)")
    print()

    for episode in tqdm(range(num_episodes)):
        scenario = scenarios[episode % 2]

        if scenario != prev_scenario and episode > 0:
            agent.exploration_rate = max(agent.exploration_rate, 0.3)
        prev_scenario = scenario

        env = SailingEnv(**get_wind_scenario(scenario))
        goal = env.goal_position.copy()

        observation, info = env.reset(seed=episode)
        agent._world_map  = None
        agent._wind_field = None

        state     = agent.discretize_state(observation)
        prev_dist = np.linalg.norm(info["position"] - goal)
        total_reward = 0

        for step in range(max_steps):
            action = agent.act(observation)
            next_obs, base_reward, done, truncated, info = env.step(action)
            next_state = agent.discretize_state(next_obs)

            curr_dist = np.linalg.norm(info["position"] - goal)
            is_stuck  = info.get("is_stuck", False)

            shaped_reward = compute_shaped_reward(
                base_reward, prev_dist, curr_dist,
                agent.discount_factor, is_stuck
            )
            prev_dist = curr_dist
            terminal  = done or truncated

            # Pass observation so learn() can trigger FNO planning
            agent.learn(state, action, shaped_reward, next_state, terminal,
                        observation=observation)

            # Update cached wind field from the new observation
            # (wind evolves each step — keep the cache fresh for FNO)
            wf_flat = next_obs[6: 6 + 128 * 128 * 2]
            agent._wind_field = wf_flat.reshape(128, 128, 2).copy()

            state       = next_state
            observation = next_obs
            total_reward += base_reward

            if terminal:
                break

        if not (done or truncated):
            agent.reset()

        rewards_history.append(total_reward)
        steps_history.append(step + 1)
        success_history.append(done)

        if episode >= WARMUP_EPISODES:
            agent.exploration_rate = max(EPSILON_FLOOR,
                                         agent.exploration_rate * DECAY_RATE)

        if (episode + 1) % 200 == 0:
            recent = success_history[-200:]
            sr = sum(recent) / len(recent) * 100
            print(
                f"Episode {episode+1:>4}: "
                f"Success (last 200): {sr:5.1f}%  |  "
                f"ε={agent.exploration_rate:.3f}  |  "
                f"Q-table: {len(agent.q_table):,}  |  "
                f"fno_buf: {len(agent.fno_buffer):,}"
            )

    print()
    print(f"Training completed!")
    print(f"  Overall success rate : {sum(success_history)/len(success_history)*100:.1f}%")
    print(f"  Q-table size         : {len(agent.q_table):,} states")

    from src.utils.agent_utils import save_qlearning_agent
    save_qlearning_agent(agent, "src/my_agent.py", agent_class_name="QLearningTrainedAgent")

    # ── Evaluation on held-out test scenario ──────────────────────────────
    print()
    print("=" * 60)
    print("Evaluating on held-out test scenario: training_3")
    print("=" * 60)

    agent.exploration_rate = 0.0
    test_env = SailingEnv(**get_wind_scenario("training_3"))

    for episode in range(5):
        observation, info = test_env.reset(seed=1000 + episode)
        agent._world_map  = None
        agent._wind_field = None
        total_reward = 0

        for step in range(500):
            action = agent.act(observation)
            observation, reward, done, truncated, info = test_env.step(action)
            wf_flat = observation[6: 6 + 128 * 128 * 2]
            agent._wind_field = wf_flat.reshape(128, 128, 2).copy()
            total_reward += reward
            if done or truncated:
                break

        print(f"Test Episode {episode+1}: Steps={step+1}, "
              f"Reward={total_reward}, Position={info['position']}, "
              f"Goal reached={done}")