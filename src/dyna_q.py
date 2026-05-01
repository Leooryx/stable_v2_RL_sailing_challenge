import numpy as np
from collections import deque
from src.agents.base_agent import BaseAgent
from src.env_sailing import SailingEnv
from src.wind_scenarios import get_wind_scenario


# =============================================================================
# FIXES APPLIED (and why each one matters)
#
# FIX 1 — Protected experience buffer (solves success collapse)
# ---------------------------------------------------------------
# The rolling deque evicted successful transitions as epsilon decayed,
# causing the agent to forget how to reach the goal (99.5% → 68%).
# Solution: split the buffer into two pools:
#   - success_buffer: stores only episodes that reached the goal (never evicted
#     until it hits its own cap of 20,000 — much later than general_buffer)
#   - general_buffer: rolling window of recent transitions
# Planning samples interleaved from both pools, so goal-reaching experience
# is always represented regardless of how deterministic the policy becomes.
#
# FIX 2 — Slower, staged epsilon decay (solves premature convergence)
# --------------------------------------------------------------------
# Original decay: 0.5 * 0.995^200 = 0.183 by episode 200.
# The agent stopped exploring while it still had a narrow, fragile policy.
# New schedule: hold epsilon=0.5 for 200 warmup episodes, then decay at
# 0.997 to a floor of 0.05 (not 0.01). Higher floor = continued exploration.
# Warm-restart epsilon=0.3 whenever the scenario switches so the agent
# re-explores under the new wind field.
#
# FIX 3 — Wind-field summary in state (solves training_3 generalization)
# -----------------------------------------------------------------------
# The original state had only local wind (wx, wy at current position).
# training_3 has a different global wind pattern, so the same local wind
# maps to completely different optimal actions depending on what lies ahead.
# Fix: sample the wind field at 4 waypoints along the direct path to goal
# and bin the dominant wind direction. This gives the agent a coarse
# "weather forecast" for its intended route without blowing up state space.
#
# FIX 4 — Prioritised planning (solves slow propagation of goal reward)
# ----------------------------------------------------------------------
# Random sampling treats a transition 1 step from goal the same as one
# 400 steps away. Fix: 20% of planning steps are drawn from success_buffer
# transitions where |reward| > 10 (near-goal or goal itself), so the +100
# signal propagates backward faster.
# =============================================================================


class MyAgent(BaseAgent):

    GRID_SIZE = 128
    GOAL = np.array([64.0, 127.0])

    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()

        self.goal_position = [64, 127]
        self.learning_rate = 0.15
        self.discount_factor = 0.995
        self.exploration_rate = 0.5

        # State discretisation
        self.position_bins = 16
        self.velocity_dir_bins = 8
        self.wind_angle_bins = 8
        self.path_wind_bins = 4      # FIX 3: coarser bins for path wind summary

        # Q-table
        self.q_table = {}

        # FIX 1: split buffer
        self.general_buffer = deque(maxlen=40_000)   # rolling recent experience
        self.success_buffer  = deque(maxlen=20_000)  # protected successful episodes
        self._episode_transitions = []               # accumulate current episode

        self.planning_steps = 50
        self.priority_fraction = 0.2                 # FIX 4

        self._world_map = None
        self._wind_field = None                      # FIX 3

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
        # Flush incomplete episode into general buffer
        for t in self._episode_transitions:
            self.general_buffer.append(t)
        self._episode_transitions = []
        self._world_map = None
        self._wind_field = None

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)

    # =========================================================================
    # State discretisation
    # =========================================================================

    def discretize_state(self, observation):
        """
        State = (x_bin, y_bin, vel_bin, local_wind_bin, path_wind_bin, danger_bin)
        """
        x,  y  = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]
        goal_x, goal_y = self.goal_position

        # Position
        x_bin = int(np.clip(x / self.GRID_SIZE * self.position_bins, 0, self.position_bins - 1))
        y_bin = int(np.clip(y / self.GRID_SIZE * self.position_bins, 0, self.position_bins - 1))

        # Velocity: direction + magnitude tier
        v_mag = np.sqrt(vx**2 + vy**2)
        if v_mag < 0.15:
            vel_bin = 0
        else:
            v_dir = np.arctan2(vy, vx)
            dir_bin = int(((v_dir + np.pi) / (2 * np.pi) * self.velocity_dir_bins)) % self.velocity_dir_bins
            mag_tier = 1 if v_mag < 0.8 else 2
            vel_bin = mag_tier * self.velocity_dir_bins + dir_bin

        # Local relative wind (wind angle vs direction-to-goal)
        dx, dy = goal_x - x, goal_y - y
        angle_to_goal = np.arctan2(dy, dx)
        wind_angle = np.arctan2(wy, wx)
        rel_wind = (wind_angle - angle_to_goal + np.pi) % (2 * np.pi) - np.pi
        local_wind_bin = int(((rel_wind + np.pi) / (2 * np.pi) * self.wind_angle_bins)) % self.wind_angle_bins

        # FIX 3: path wind summary
        path_wind_bin = self._path_wind_bin(x, y, goal_x, goal_y)

        # Danger lookahead
        danger_bin = 0
        if v_mag > 0.1 and self._world_map is not None:
            look_x = int(np.clip(x + vx * 2, 0, self.GRID_SIZE - 1))
            look_y = int(np.clip(y + vy * 2, 0, self.GRID_SIZE - 1))
            if self._world_map[look_y, look_x] == 1:
                danger_bin = 1

        return (x_bin, y_bin, vel_bin, local_wind_bin, path_wind_bin, danger_bin)

    def _path_wind_bin(self, x, y, goal_x, goal_y):
        """
        Sample wind at 4 waypoints along the straight-line path to goal.
        Returns the binned average wind direction across those points.
        """
        if self._wind_field is None:
            return 0

        avg_wx, avg_wy = 0.0, 0.0
        for t in [0.25, 0.5, 0.75, 1.0]:
            px = int(np.clip(x + t * (goal_x - x), 0, self.GRID_SIZE - 1))
            py = int(np.clip(y + t * (goal_y - y), 0, self.GRID_SIZE - 1))
            avg_wx += self._wind_field[py, px, 0]
            avg_wy += self._wind_field[py, px, 1]

        avg_angle = np.arctan2(avg_wy, avg_wx)
        return int(((avg_angle + np.pi) / (2 * np.pi) * self.path_wind_bins)) % self.path_wind_bins

    # =========================================================================
    # Learning
    # =========================================================================

    def learn(self, state, action, shaped_reward, next_state, terminal=False):
        # Real Q-update
        self._q_update(state, action, shaped_reward, next_state, terminal)

        # Accumulate episode transitions
        self._episode_transitions.append(
            (state, action, shaped_reward, next_state, terminal)
        )

        # FIX 1: on episode end, route to correct buffer
        if terminal:
            goal_reached = (shaped_reward > 50)
            target = self.success_buffer if goal_reached else self.general_buffer
            for t in self._episode_transitions:
                target.append(t)
            self._episode_transitions = []

        # Planning
        total = len(self.general_buffer) + len(self.success_buffer)
        if total < 10:
            return

        n_priority = int(self.planning_steps * self.priority_fraction)
        n_uniform  = self.planning_steps - n_priority

        self._plan_uniform(n_uniform)
        self._plan_priority(n_priority)

    def _plan_uniform(self, n):
        buffers = []
        if self.general_buffer: buffers.append(self.general_buffer)
        if self.success_buffer:  buffers.append(self.success_buffer)
        if not buffers:
            return
        for i in range(n):
            buf = buffers[i % len(buffers)]
            idx = int(self.np_random.integers(0, len(buf)))
            s, a, r, sn, term = buf[idx]
            self._q_update(s, a, r, sn, term)

    def _plan_priority(self, n):
        if not self.success_buffer or n == 0:
            return
        high_reward = [(s, a, r, sn, t) for s, a, r, sn, t in self.success_buffer
                       if abs(r) > 10]
        source = high_reward if high_reward else list(self.success_buffer)
        indices = self.np_random.integers(0, len(source), size=n)
        for idx in indices:
            s, a, r, sn, term = source[idx]
            self._q_update(s, a, r, sn, term)

    def _q_update(self, state, action, reward, next_state, terminal):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(9)
        td_target = (reward if terminal
                     else reward + self.discount_factor * np.max(self.q_table[next_state]))
        self.q_table[state][action] += self.learning_rate * (
            td_target - self.q_table[state][action]
        )

    # =========================================================================
    # Default / fallback policy
    # =========================================================================

    def default_policy(self, obs):
        dx = self.goal_position[0] - obs[0]
        dy = self.goal_position[1] - obs[1]
        angle = np.arctan2(dy, dx)
        return int(((angle + np.pi) / (2 * np.pi)) * 9) % 9

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _extract_maps(self, observation):
        wind_flat = observation[6: 6 + self.GRID_SIZE * self.GRID_SIZE * 2]
        self._wind_field = wind_flat.reshape(self.GRID_SIZE, self.GRID_SIZE, 2)
        map_start = 6 + self.GRID_SIZE * self.GRID_SIZE * 2
        flat_map = observation[map_start: map_start + self.GRID_SIZE * self.GRID_SIZE]
        self._world_map = flat_map.reshape(self.GRID_SIZE, self.GRID_SIZE)


# =============================================================================
# Training pipeline
# =============================================================================

def compute_shaped_reward(base_reward, prev_dist, curr_dist, discount_factor, is_stuck):
    phi_next = -curr_dist
    phi_curr = -prev_dist
    shaping = (discount_factor * phi_next) - phi_curr
    shaped = base_reward + shaping * 0.2
    if is_stuck:
        shaped -= 50.0
    return shaped


if __name__ == "__main__":

    agent = MyAgent()
    np.random.seed(42)
    agent.seed(42)

    scenarios = ['training_1', 'training_2']
    max_steps = 500
    num_episodes = 2000

    rewards_history = []
    steps_history = []
    success_history = []

    print("Starting Dyna-Q training (v2)...")
    print(f"  Planning steps per real step : {agent.planning_steps}")
    print(f"  Priority fraction            : {agent.priority_fraction}")
    print(f"  Episodes                     : {num_episodes}")
    print()

    # FIX 2: staged epsilon schedule
    WARMUP_EPISODES = 200
    DECAY_RATE = 0.997
    EPSILON_FLOOR = 0.05

    prev_scenario = None

    for episode in range(num_episodes):
        scenario = scenarios[episode % 2]

        # FIX 2: warm-restart on scenario switch
        if scenario != prev_scenario and episode > 0:
            agent.exploration_rate = max(agent.exploration_rate, 0.3)
        prev_scenario = scenario

        env = SailingEnv(**get_wind_scenario(scenario))
        goal = env.goal_position.copy()

        observation, info = env.reset(seed=episode)
        agent._world_map = None
        agent._wind_field = None

        state = agent.discretize_state(observation)
        prev_dist = np.linalg.norm(info['position'] - goal)
        total_reward = 0

        for step in range(max_steps):
            action = agent.act(observation)
            next_observation, base_reward, done, truncated, info = env.step(action)
            next_state = agent.discretize_state(next_observation)

            curr_dist = np.linalg.norm(info['position'] - goal)
            is_stuck = info.get('is_stuck', False)

            shaped_reward = compute_shaped_reward(
                base_reward, prev_dist, curr_dist,
                agent.discount_factor, is_stuck
            )
            prev_dist = curr_dist

            terminal = done or truncated
            agent.learn(state, action, shaped_reward, next_state, terminal)

            state = next_state
            observation = next_observation
            total_reward += base_reward

            if terminal:
                break

        # Flush buffer if episode ended by truncation without terminal signal
        if not (done or truncated):
            agent.reset()

        rewards_history.append(total_reward)
        steps_history.append(step + 1)
        success_history.append(done)

        # FIX 2: staged decay
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
                f"succ_buf: {len(agent.success_buffer):,}  |  "
                f"gen_buf: {len(agent.general_buffer):,}"
            )

    print()
    print(f"Training completed!")
    print(f"  Overall success rate  : {sum(success_history)/len(success_history)*100:.1f}%")
    print(f"  Q-table size          : {len(agent.q_table):,} states")
    print(f"  Success buffer        : {len(agent.success_buffer):,} transitions")
    print(f"  General buffer        : {len(agent.general_buffer):,} transitions")

    from src.utils.agent_utils import save_qlearning_agent
    output_path = 'src/my_agent.py'
    save_qlearning_agent(agent, output_path, agent_class_name="QLearningTrainedAgent")

    # =========================================================================
    # Evaluation — training_3 is the held-out test scenario
    # =========================================================================
    print()
    print("=" * 60)
    print("Evaluating on held-out test scenario: training_3")
    print("=" * 60)

    agent.exploration_rate = 0.0

    test_env = SailingEnv(**get_wind_scenario('training_3'))
    goal = test_env.goal_position.copy()

    print("Testing the trained agent on 5 new episodes...")
    for episode in range(5):
        observation, info = test_env.reset(seed=1000 + episode)
        agent._world_map = None
        agent._wind_field = None
        total_reward = 0

        for step in range(500):
            action = agent.act(observation)
            observation, reward, done, truncated, info = test_env.step(action)
            total_reward += reward
            if done or truncated:
                break

        print(
            f"Test Episode {episode+1}: "
            f"Steps={step+1}, "
            f"Reward={total_reward}, "
            f"Position={info['position']}, "
            f"Goal reached={done}"
        )