import kagglehub 
# Download latest version 
path = kagglehub.dataset_download("coplin/traffic") 
print("Path to dataset files:", path)


import os
import glob
import random
from collections import namedtuple, deque
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium import spaces

# Data loading(baseline)

def load_kaggle_traffic_csvs(data_dir):
    """Load all CSV files in data_dir; expects each CSV to have columns 'time' and 'count'.
    Returns DataFrame with columns ['sensor','time','count'].
    """
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if len(files) == 0:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = []
    for f in sorted(files):
        df = pd.read_csv(f)
        # adapt to different column naming conventions
        if 'time' not in df.columns or 'count' not in df.columns:
            cols = df.columns.tolist()
            if len(cols) >= 2:
                df = df.iloc[:, :2].copy()
                df.columns = ['time', 'count']
            else:
                raise ValueError(f"CSV {f} does not have expected columns")
        # ensure datetime
        try:
            df['time'] = pd.to_datetime(df['time'])
        except Exception:
            df['time'] = pd.to_datetime(df['time'], unit='s')
        sensor_name = os.path.splitext(os.path.basename(f))[0]
        df['sensor'] = sensor_name
        dfs.append(df[['sensor', 'time', 'count']].copy())

    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.sort_values(['sensor', 'time']).reset_index(drop=True)
    return all_df

def build_hourly_baselines(df):
    """Return baseline average counts per sensor per hour (0-23)."""
    df = df.copy()
    df['hour'] = df['time'].dt.hour
    baseline = df.groupby(['sensor', 'hour'])['count'].mean().reset_index()
    baseline.rename(columns={'count': 'avg_count'}, inplace=True)
    return baseline

def build_hourly_baseline_map(baseline_df, num_segments):
    """
    Map sensor-level hourly averages onto environment segments.
    Approach:
      - Sort sensors consistently, place their sensor indices evenly across the segments.
      - If sensors < segments, linearly interpolate between sensor positions.
      - Result: baseline_map[hour] = {seg_idx: avg_count_for_segment}
    """
    sensors = sorted(baseline_df['sensor'].unique())
    n_sensors = len(sensors)
    if n_sensors == 0:
        # fallback to zeros
        return {h: {i: 0.0 for i in range(num_segments)} for h in range(24)}

    # build sensor->position along [0, num_segments)
    sensor_positions = np.linspace(0, num_segments - 1, num=n_sensors)

    # produce per-sensor hourly arrays
    sensor_hour_matrix = np.zeros((n_sensors, 24), dtype=np.float32)
    sensor_to_idx = {s: i for i, s in enumerate(sensors)}
    hour_group = baseline_df.groupby(['sensor', 'hour'])['avg_count'].mean()

    # fill sensor_hour_matrix; if a sensor-hour missing fill with sensor mean or global
    global_mean = baseline_df['avg_count'].mean() if len(baseline_df) > 0 else 0.0
    for s in sensors:
        idx = sensor_to_idx[s]
        row = hour_group.loc[s] if s in hour_group.index.get_level_values(0) else None
        for h in range(24):
            if (s, h) in hour_group.index:
                sensor_hour_matrix[idx, h] = float(hour_group.loc[(s, h)])
            else:
                # fallback to sensor mean if available, else global mean
                sensor_vals = baseline_df[baseline_df['sensor'] == s]['avg_count']
                sensor_mean = float(sensor_vals.mean()) if len(sensor_vals) > 0 else global_mean
                sensor_hour_matrix[idx, h] = sensor_mean

    # For each hour, interpolate across segments
    baseline_map = {}
    seg_positions = np.arange(num_segments)
    # for numerical stability if n_sensors==1, broadcast
    if n_sensors == 1:
        for h in range(24):
            val = float(sensor_hour_matrix[0, h])
            baseline_map[h] = {i: val for i in range(num_segments)}
        return baseline_map

    # interpolate sensor_hour_matrix across segment positions
    for h in range(24):
        vals = sensor_hour_matrix[:, h]
        # linear interpolation of sensor values to each segment position
        interp = np.interp(seg_positions, sensor_positions, vals).astype(np.float32)
        baseline_map[h] = {int(i): float(interp[i]) for i in range(num_segments)}

    return baseline_map

# ---------------------------
# Plot helpers (kept + small fixes)
# ---------------------------
def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_hourly_model_vs_baseline(metrics_hourly_path, baseline_df, save_dir='runs_ddqn_fixed'):
    if not os.path.exists(metrics_hourly_path):
        print("Hourly metrics file not found:", metrics_hourly_path)
        return
    hourly_model = pd.read_csv(metrics_hourly_path)
    baseline_mean = baseline_df.groupby('hour')['avg_count'].mean().reset_index()
    _ensure_dir(save_dir)

    plt.figure(figsize=(10,6))
    plt.plot(baseline_mean['hour'], baseline_mean['avg_count'], label='Baseline Avg (sensors mean)', marker='o')
    plt.plot(hourly_model['hour'], hourly_model['model_avg_density'], label='Model Avg Density', marker='x')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Density per Segment')
    plt.title('Hourly Model vs Baseline Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'baseline_vs_model_hourly_density.png'))
    plt.close()

def plot_training_metrics(metrics_csv_path, baseline_df=None, save_dir=None):
    if not os.path.exists(metrics_csv_path):
        print("metrics file not found:", metrics_csv_path)
        return
    df = pd.read_csv(metrics_csv_path)
    if save_dir is None:
        save_dir = os.path.dirname(metrics_csv_path)
    _ensure_dir(save_dir)

    def safe_plot(x, y, title, fname, xlabel='Episode', ylabel=None):
        plt.figure(figsize=(8,5))
        plt.plot(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel or title)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, fname))
        plt.close()

    safe_plot(df['episode'], df['episode_reward'], 'Episode Reward Over Time', 'episode_reward.png')
    safe_plot(df['episode'], df['avg_reward_per_agent'], 'Avg Reward per Agent', 'avg_agent_reward.png')
    safe_plot(df['episode'], df['loss'], 'Loss', 'loss.png')
    safe_plot(df['episode'], df['collision_count'], 'Collisions per Episode', 'collisions.png')
    safe_plot(df['episode'], df['avg_density'], 'Avg Density', 'avg_density.png')

    if baseline_df is not None:
        baseline_hourly = baseline_df.groupby('hour')['avg_count'].mean()
        model_mean_density = df['avg_density'].mean()
        plt.figure(figsize=(8,5))
        plt.plot(baseline_hourly.index, baseline_hourly.values, label='Baseline Avg Density', marker='o')
        plt.hlines(model_mean_density, xmin=0, xmax=23, linestyles='--', label='Model Mean Density')
        plt.xlabel('Hour of Day'); plt.ylabel('Density (avg)'); plt.title('Model vs Baseline Density')
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'baseline_vs_model_density.png'))
        plt.close()

    print("✅ Plots saved in:", save_dir)

def summarize_results(metrics_csv_path, baseline_df=None):
    df = pd.read_csv(metrics_csv_path)
    print("\n=== Training Summary ===")
    print(f"Episodes: {len(df)}")
    print(f"Final Avg Reward per Agent: {df['avg_reward_per_agent'].iloc[-1]:.3f}")
    print(f"Final Loss: {df['loss'].iloc[-1]:.6f}")
    print(f"Mean Collisions per Episode: {df['collision_count'].mean():.3f}")
    print(f"Mean Density: {df['avg_density'].mean():.3f}")
    if baseline_df is not None:
        baseline_mean = baseline_df['avg_count'].mean()
        print(f"Baseline Mean Density (sensors): {baseline_mean:.3f}")
        diff = df['avg_density'].mean() - baseline_mean
        print(f"Δ Model vs Baseline Density: {diff:+.3f}")
    print("=========================\n")

# ---------------------------
# Environment
# ---------------------------

class TrafficEnv(AECEnv):
    """Multi-agent traffic environment with local neighborhood awareness and collision avoidance.
    - Agents observe local info only: own_speed, dist_to_next, local_density, left_density, right_density
    - No explicit communication between agents, shared policy allowed.
    """

    metadata = {"render.modes": ["human"], "name": "traffic_env"}

    def __init__(self, n_agents=80, num_segments=50, max_speed=5.0, dt=1.0, baseline_counts=None, neighbor_radius=1, seed=None):
        super().__init__()
        self.n_agents = int(n_agents)
        self.num_segments = int(num_segments)
        self.max_speed = float(max_speed)
        self.dt = float(dt)
        self.neighbor_radius = int(neighbor_radius)
        self.baseline_counts = baseline_counts  # dict mapping segment idx -> baseline density (or None)

        # agents list & state arrays
        self.agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.pos = np.zeros(self.n_agents, dtype=np.float32)
        self.speed = np.zeros(self.n_agents, dtype=np.float32)

        # action & observation spaces
        self.action_spaces = {a: spaces.Discrete(3) for a in self.agents}
        high = np.array([self.max_speed, self.num_segments, 1e6, 1e6, 1e6], dtype=np.float32)
        low = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.observation_spaces = {a: spaces.Box(low=low, high=high, dtype=np.float32) for a in self.agents}

        # AEC bookkeeping
        self._agent_selector = agent_selector(self.agents)
        self.current_agent = None
        self.rewards = {a: 0.0 for a in self.agents}
        self.dones = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._cumulative_steps = 0
        self._steps_since_reset = 0

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.reset()

    def reset(self, seed=None, return_info=False, options=None):
        # even spacing with small jitter prevents start collisions
        base_positions = np.linspace(0, self.num_segments, num=self.n_agents, endpoint=False)
        avg_spacing = max(1.0, self.num_segments / max(1.0, self.n_agents))
        jitter = np.random.uniform(-0.02 * avg_spacing, 0.02 * avg_spacing, size=self.n_agents)
        self.pos = (base_positions + jitter).astype(np.float32) % self.num_segments

        # safe initial speeds
        self.speed = np.random.uniform(self.max_speed * 0.3, self.max_speed * 0.5, size=self.n_agents).astype(np.float32)

        # reset bookkeeping
        self._agent_selector = agent_selector(self.agents); self._agent_selector.reset()
        self.current_agent = self._agent_selector.next()
        self.rewards = {a: 0.0 for a in self.agents}
        self.dones = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._cumulative_steps = 0
        self._steps_since_reset = 0

        self._update_segment_density()
        return self._observe(self.current_agent)[0]

    def update_hour(self, hour, baseline_map):
        if baseline_map is None:
            return
        if hour in baseline_map:
            self.baseline_counts = baseline_map[hour]
        self._update_segment_density()

    def _update_segment_density(self):
        """Compute occupancy per segment from agent positions, blend with baseline if available."""
        seg = np.zeros(self.num_segments, dtype=np.float32)
        for i in range(self.n_agents):
            sidx = int(self.pos[i]) % self.num_segments
            seg[sidx] += 1.0
        if self.baseline_counts is not None:
            # if baseline provided for this hour, blend per-segment baseline (0.5 simulated + 0.5 baseline)
            for seg_idx, val in self.baseline_counts.items():
                if 0 <= seg_idx < self.num_segments:
                    seg_val = seg[seg_idx]
                    seg[seg_idx] = 0.5 * seg_val + 0.5 * float(val)
        self.segment_density = seg

    def _observe(self, agent):
        """Return (obs, reward_placeholder, done_placeholder, info_placeholder) for compatibility."""
        idx = int(agent.split('_')[1])
        own_speed = float(self.speed[idx])
        # distance to next ahead in circular road; positive distance
        diffs = (self.pos - self.pos[idx]) % self.num_segments
        diffs[idx] = np.inf
        dist_to_next = float(np.min(diffs))

        seg = int(self.pos[idx]) % self.num_segments
        local_baseline = float(self.segment_density[seg])
        left_seg = (seg - self.neighbor_radius) % self.num_segments
        right_seg = (seg + self.neighbor_radius) % self.num_segments
        left_density = float(self.segment_density[left_seg])
        right_density = float(self.segment_density[right_seg])

        obs = np.array([own_speed, dist_to_next, local_baseline, left_density, right_density], dtype=np.float32)
        return obs, 0.0, False, {}

    def observe(self, agent):
        obs, _, _, _ = self._observe(agent)
        return obs

    def step(self, action):
        """AEC single-agent step for compatibility (we still use step_all_agents in training)."""
        agent = self.current_agent
        idx = int(agent.split('_')[1])
        if self.dones[agent]:
            self.current_agent = self._agent_selector.next()
            return

        accel = float(action - 1.0)  # -1,0,+1
        self.speed[idx] = np.clip(self.speed[idx] + accel * 0.4, 0.0, self.max_speed)
        self.pos[idx] = (self.pos[idx] + self.speed[idx] * self.dt) % self.num_segments

        self._update_segment_density()

        # compute local metrics
        pos_i = self.pos[idx]
        rel = (self.pos - pos_i + self.num_segments / 2.0) % self.num_segments - self.num_segments / 2.0
        local_count = float(np.sum(np.abs(rel) < 1.0))
        density_norm = min(1.0, local_count / max(1.0, self.n_agents))

        diffs = (self.pos - pos_i) % self.num_segments
        diffs[idx] = np.inf
        dist_to_next = float(np.min(diffs))

        # collision detection & penalty (consistent, smooth)
        collision_penalty = 0.0
        collision_happened = False
        if self._steps_since_reset > 5:
            if dist_to_next < 0.2:
                collision_happened = True
                collision_penalty = -1.5 * (1.0 - dist_to_next / 0.2)  # strong short-range penalty
            elif dist_to_next < 0.6:
                collision_penalty = -0.5 * (0.6 - dist_to_next) / 0.6  # soft nudge

        # reward: encourage safe speed & discourage high local density & large accelerations
        safe_speed = self.speed[idx] if dist_to_next > 0.5 else 0.0
        accel_cost = 0.02 * abs(action - 1.0)
        reward = 0.9 * (safe_speed / max(1e-6, self.max_speed)) - 0.25 * density_norm - accel_cost + collision_penalty
        reward = float(np.clip(reward, -3.0, 3.0))

        self.rewards[agent] = reward
        self._cumulative_steps += 1
        self._steps_since_reset += 1

        done = (self._cumulative_steps >= 1000)
        self.dones[agent] = done
        self.infos[agent] = {'local_density': float(local_count), 'dist_to_next': float(dist_to_next), 'collision': collision_happened}

        # advance agent selector safely
        try:
            self.current_agent = self._agent_selector.next()
        except Exception:
            self._agent_selector.reset()
            self.current_agent = self._agent_selector.next()

    def last(self):
        agent = self.current_agent
        obs = self.observe(agent)
        return obs, self.rewards[agent], self.dones[agent], self.infos[agent]

    def step_all_agents(self, actions_dict):
        """Parallel step: actions_dict: agent -> action"""
        # apply dynamics
        for a, act in actions_dict.items():
            idx = int(a.split('_')[1])
            accel = float(act - 1.0)
            self.speed[idx] = np.clip(self.speed[idx] + accel * 0.4, 0.0, self.max_speed)
            self.pos[idx] = (self.pos[idx] + self.speed[idx] * self.dt) % self.num_segments

        # update densities once
        self._update_segment_density()

        rewards = {}
        collision_count = 0
        # compute rewards per-agent
        for i, a in enumerate(self.agents):
            pos_i = self.pos[i]
            rel = (self.pos - pos_i + self.num_segments / 2.0) % self.num_segments - self.num_segments / 2.0
            local_count = float(np.sum(np.abs(rel) < 1.0))
            density_norm = min(1.0, local_count / max(1.0, self.n_agents))

            diffs = (self.pos - pos_i) % self.num_segments
            diffs[i] = np.inf
            dist_to_next = float(np.min(diffs))

            collision_penalty = 0.0
            collision_happened = False
            if self._steps_since_reset > 5:
                if dist_to_next < 0.2:
                    collision_happened = True
                    collision_penalty = -1.5 * (1.0 - dist_to_next / 0.2)
                elif dist_to_next < 0.6:
                    collision_penalty = -0.5 * (0.6 - dist_to_next) / 0.6

            safe_speed = self.speed[i] if dist_to_next > 0.5 else 0.0
            accel_cost = 0.02 * abs(actions_dict[a] - 1.0)
            r = 0.9 * (safe_speed / max(1e-6, self.max_speed)) - 0.25 * density_norm - accel_cost + collision_penalty
            r = float(np.clip(r, -3.0, 3.0))

            rewards[a] = r
            self.rewards[a] = r
            self.dones[a] = False
            self.infos[a] = {'local_density': float(local_count), 'dist_to_next': float(dist_to_next), 'collision': collision_happened}
            if collision_happened:
                collision_count += 1

        self._cumulative_steps += 1
        self._steps_since_reset += 1
        if self._cumulative_steps >= 1000:
            for a in self.agents:
                self.dones[a] = True

        return rewards

    def render(self, mode='human'):
        segs = ['.' for _ in range(self.num_segments)]
        for i in range(self.n_agents):
            p = int(self.pos[i]) % self.num_segments
            segs[p] = 'V'
        print(''.join(segs))


# ---------------------------
# Agent & training utils
# ---------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import numpy as np
import random, os, pandas as pd

# --- PPO Network ---
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def act(self, obs):
        logits, value = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

# --- Rollout Buffer for PPO ---
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states, self.actions, self.logprobs, self.rewards, self.dones, self.values = [], [], [], [], [], []

    def add(self, s, a, lp, r, d, v):
        self.states.append(s)
        self.actions.append(a)
        self.logprobs.append(lp)
        self.rewards.append(r)
        self.dones.append(d)
        self.values.append(v)

# --- Observation Normalization ---
def normalize_obs(obs, max_speed, num_segments, max_density):
    o = np.array(obs, dtype=np.float32)
    o[0] = o[0] / max(1e-6, max_speed)
    o[1] = o[1] / max(1.0, num_segments)
    if o.shape[0] > 2:
        o[2:] = o[2:] / max(1.0, max_density)
    return o

# --- PPO Update ---
def ppo_update(actor_critic, optimizer, buffer, gamma=0.99, lam=0.95, clip_eps=0.2, epochs=4, batch_size=128, device='cpu'):
    states = torch.tensor(np.array(buffer.states), dtype=torch.float32, device=device)
    actions = torch.tensor(buffer.actions, dtype=torch.long, device=device)
    old_logprobs = torch.tensor(buffer.logprobs, dtype=torch.float32, device=device)
    rewards = np.array(buffer.rewards, dtype=np.float32)
    dones = np.array(buffer.dones, dtype=np.float32)
    values = np.array([v.item() for v in buffer.values], dtype=np.float32)

    # Compute advantages with GAE
    advantages, returns = [], []
    gae = 0
    next_value = 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
        next_value = values[step]
    advantages = np.array(advantages, dtype=np.float32)
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Mini-batch PPO update
    dataset_size = len(states)
    for _ in range(epochs):
        idx = np.arange(dataset_size)
        np.random.shuffle(idx)
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            batch_idx = idx[start:end]
            b_states = states[batch_idx]
            b_actions = actions[batch_idx]
            b_oldlog = old_logprobs[batch_idx]
            b_adv = torch.tensor(advantages[batch_idx], dtype=torch.float32, device=device)
            b_returns = torch.tensor(returns[batch_idx], dtype=torch.float32, device=device)

            logits, values_new = actor_critic(b_states)
            dist = torch.distributions.Categorical(F.softmax(logits, dim=-1))
            new_logprobs = dist.log_prob(b_actions)

            ratio = torch.exp(new_logprobs - b_oldlog)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values_new.squeeze(), b_returns)
            loss = actor_loss + 0.5 * critic_loss - 0.001 * dist.entropy().mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor_critic.parameters(), 1.0)
            optimizer.step()

# --- Training Loop ---
def train_ppo(env, baseline_map, device='cpu', episodes=200, steps_per_hour=25,
              gamma=0.99, lr=3e-4, lam=0.95,
              clip_eps=0.2, epochs=4, batch_size=128,
              eps_start=1.0, eps_final=0.1, eps_decay_episodes=300,
              save_dir='runs_ppo_fixed', seed=None):
    os.makedirs(save_dir, exist_ok=True)
    if seed is not None:
        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)

    obs_dim = env.observation_spaces[env.agents[0]].shape[0]
    n_actions = env.action_spaces[env.agents[0]].n
    model = ActorCritic(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    metrics = {'episode': [], 'episode_reward': [], 'avg_reward_per_agent': [], 'loss': [], 'avg_density': [], 'collision_count': []}
    hourly_totals = np.zeros(24, dtype=np.float64)

    overall_max_baseline = 0.0
    if baseline_map is not None and len(baseline_map) > 0:
        overall_max_baseline = max([max(seg_map.values()) for seg_map in baseline_map.values()])
    max_density = max(overall_max_baseline, env.n_agents / max(1, env.num_segments), 1.0)

    buffer = RolloutBuffer()

    for ep in range(episodes):
        eps = max(eps_final, eps_start - (eps_start - eps_final) * (ep / max(1.0, eps_decay_episodes)))
        env.reset()
        ep_reward = 0.0
        agent_rewards = {a: 0.0 for a in env.agents}
        density_accum, collision_count, frames = 0.0, 0, 0
        hour_density_accum = np.zeros(24, dtype=np.float64)
        hour_steps_count = np.zeros(24, dtype=np.int32)
        buffer.clear()

        for hour in range(24):
            env.update_hour(hour, baseline_map)
            for t in range(steps_per_hour):
                actions = {}
                logps = {}
                values = {}
                states_cache = {}

                for a in env.agents:
                    s_raw = env.observe(a)
                    s = normalize_obs(s_raw, env.max_speed, env.num_segments, max_density)
                    states_cache[a] = s

                    st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                    if random.random() < eps:
                        act = random.randrange(n_actions)
                        lp = torch.tensor(0.0, device=device)
                        v = torch.tensor(0.0, device=device)
                    else:
                        act, lp, v = model.act(st)
                    actions[a] = act
                    logps[a] = lp
                    values[a] = v

                rewards_dict = env.step_all_agents(actions)

                for a in env.agents:
                    s = states_cache[a]
                    a_act = actions[a]
                    lp = logps[a]
                    v = values[a]
                    r = float(np.clip(float(rewards_dict[a]), -5.0, 5.0))
                    s_next_raw = env.observe(a)
                    done = bool(env.dones[a])
                    buffer.add(s, a_act, lp.item(), r, done, v)

                    ep_reward += r
                    agent_rewards[a] += r
                    if s_next_raw[1] < 0.2:
                        collision_count += 1

                density_accum += float(np.mean(env.segment_density))
                hour_density_accum[hour] += float(np.mean(env.segment_density))
                hour_steps_count[hour] += 1
                frames += 1

                if all(env.dones.values()):
                    break

        # PPO update after each episode
        ppo_update(model, optimizer, buffer, gamma=gamma, lam=lam, clip_eps=clip_eps, epochs=epochs, batch_size=batch_size, device=device)

        mean_agent_reward = float(np.mean(list(agent_rewards.values())))
        avg_density = float(density_accum / max(1, frames))
        metrics['episode'].append(ep)
        metrics['episode_reward'].append(ep_reward)
        metrics['avg_reward_per_agent'].append(mean_agent_reward)
        metrics['loss'].append(0.0)
        metrics['avg_density'].append(avg_density)
        agent_steps = frames * env.n_agents
        collision_rate = float(collision_count) / max(1.0, agent_steps)
        metrics['collision_count'].append(collision_rate)

        for h in range(24):
            if hour_steps_count[h] > 0:
                hourly_totals[h] += (hour_density_accum[h] / hour_steps_count[h])

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"Ep {ep+1}/{episodes} | EpReward {ep_reward:.2f} | AvgAgent {mean_agent_reward:.3f} | Eps {eps:.4f} | AvgDensity {avg_density:.3f} | Collisions {collision_rate:.4f}")

        if (ep + 1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"ppo_ep{ep+1}.pth"))
            pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "metrics_partial.csv"), index=False)

    torch.save(model.state_dict(), os.path.join(save_dir, "ppo_final.pth"))
    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "metrics_final.csv"), index=False)
    hourly_model_avg = hourly_totals / max(1, episodes)
    pd.DataFrame({'hour': np.arange(24), 'model_avg_density': hourly_model_avg}).to_csv(os.path.join(save_dir, "metrics_hourly.csv"), index=False)
    return model, metrics


class Args:
    def __init__(self):
        self.data_dir = '/kaggle/input/traffic/DataSet/'   # path to dataset
        self.num_agents = 80
        self.num_segments = 50

        # Training hyperparameters
        self.episodes = 200
        self.steps_per_hour = 100
        self.gamma = 0.99
        self.lr = 3e-4
        self.lam = 0.95
        self.clip_eps = 0.2
        self.epochs = 4
        self.batch_size = 128

        # Exploration schedule
        self.eps_start = 1.0
        self.eps_final = 0.05
        self.eps_decay_episodes = 300

        # Saving and reproducibility
        self.save_dir = 'runs_ppo_fixed'
        self.seed = 1234


# === Main run ===
if __name__ == "__main__":
    args = Args()
    print("Loading dataset from:", args.data_dir)

    # --- Load baseline traffic data ---
    all_df = load_kaggle_traffic_csvs(args.data_dir)
    baseline_df = build_hourly_baselines(all_df)
    baseline_map = build_hourly_baseline_map(baseline_df, args.num_segments)

    # --- Initialize environment ---
    env = TrafficEnv(
        n_agents=args.num_agents,
        num_segments=args.num_segments,
        baseline_counts=None,
        neighbor_radius=1,
        seed=args.seed
    )

    # --- Select device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # --- PPO Training ---
    model, metrics = train_ppo(
        env=env,
        baseline_map=baseline_map,
        device=device,

        # Training schedule
        episodes=args.episodes,
        steps_per_hour=args.steps_per_hour,
        gamma=args.gamma,
        lam=args.lam,
        lr=args.lr,
        clip_eps=args.clip_eps,
        epochs=args.epochs,
        batch_size=args.batch_size,

        # Exploration
        eps_start=args.eps_start,
        eps_final=args.eps_final,
        eps_decay_episodes=args.eps_decay_episodes,

        # Saving and reproducibility
        save_dir=args.save_dir,
        seed=args.seed
    )

    metrics_path = os.path.join(args.save_dir, "metrics_final.csv")
    hourly_path = os.path.join(args.save_dir, "metrics_hourly.csv")

    if os.path.exists(metrics_path):
        plot_training_metrics(metrics_path, baseline_df=baseline_df, save_dir=args.save_dir)
        summarize_results(metrics_path, baseline_df=baseline_df)
    else:
        print(f"[Warning] Metrics file not found at {metrics_path}")

    if os.path.exists(hourly_path):
        plot_hourly_model_vs_baseline(hourly_path, baseline_df, save_dir=args.save_dir)
    else:
        print(f"[Warning] Hourly metrics file not found at {hourly_path}")



