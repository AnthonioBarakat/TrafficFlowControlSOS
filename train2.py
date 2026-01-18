# train2.py
"""
Dueling Double DQN trainer for the TrafficEnv (shared policy across agents).

Usage:
    python train2.py --data_dir /path/to/csvs --episodes 800 --num_agents 80 --num_segments 50

Requirements:
    pip install torch numpy pandas tqdm

Notes:
    - Expects your TrafficEnv class (the enhanced version) to be importable from env.py
    - Expects load_kaggle_traffic_csvs and build_hourly_baselines to be available (same module or importable)
"""
import argparse
import os
import random
from collections import deque, namedtuple
import numpy as np
import pandas as pd
from tqdm import trange, tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Import your environment and data loader functions (make sure env.py is in same folder or in PYTHONPATH)
from env import TrafficEnv
from data_loader import load_kaggle_traffic_csvs, build_hourly_baselines

# If your project keeps everything in one file, adjust imports accordingly.
# For this script I'll assume `TrafficEnv`, `load_kaggle_traffic_csvs`, `build_hourly_baselines` are available in the current namespace.


### -------------------
### Replay buffer
### -------------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


### -------------------
### Dueling Q-network
### -------------------
class DuelingQNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        # shared
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

        # value stream
        self.value_fc = nn.Linear(hidden, hidden // 2)
        self.value_out = nn.Linear(hidden // 2, 1)

        # advantage stream
        self.adv_fc = nn.Linear(hidden, hidden // 2)
        self.adv_out = nn.Linear(hidden // 2, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        v = F.relu(self.value_fc(x))
        v = self.value_out(v)                       # shape: (batch, 1)

        a = F.relu(self.adv_fc(x))
        a = self.adv_out(a)                         # shape: (batch, n_actions)

        # combine: Q = V + (A - mean(A))
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


### -------------------
### Utilities: baseline mapping
### -------------------
def build_hourly_baseline_map(baseline_df, num_segments):
    """
    baseline_df: DataFrame with columns ['sensor', 'hour', 'avg_count'] from build_hourly_baselines
    Returns: dict hour -> dict(segment_idx -> value)
    Approach:
      - For each hour (0..23), compute mean across sensors: baseline_mean_hour[hour]
      - Map that scalar to segments by tiling to length num_segments (this is simple but consistent)
    """
    # ensure baseline_df has required columns
    if not {'sensor', 'hour', 'avg_count'}.issubset(set(baseline_df.columns)):
        raise ValueError("baseline_df must have columns ['sensor','hour','avg_count']")

    hour_mean = baseline_df.groupby('hour')['avg_count'].mean()
    baseline_map = {}
    for hour in range(24):
        if hour in hour_mean.index:
            val = float(hour_mean.loc[hour])
        else:
            val = float(hour_mean.mean())
        # map to segments by repeating (tile)
        arr = np.tile(val, num_segments)
        seg_map = {i: float(arr[i]) for i in range(num_segments)}
        baseline_map[hour] = seg_map
    return baseline_map


### -------------------
### Double DQN update helper
### -------------------
def ddqn_update(online_net, target_net, optimizer, batch, gamma=0.99, device='cpu'):
    """
    Performs a Double DQN update given a batch (Transition tuple).
    Returns MSE loss value (float).
    """
    states = torch.tensor(np.stack(batch.state), dtype=torch.float32, device=device)  # (B, obs_dim)
    actions = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)  # (B,1)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)  # (B,1)
    next_states = torch.tensor(np.stack(batch.next_state), dtype=torch.float32, device=device)
    dones = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

    # Current Q estimates
    q_values = online_net(states).gather(1, actions)  # (B,1)

    # Double DQN target:
    # next actions by online_net, value estimated by target_net
    with torch.no_grad():
        next_q_online = online_net(next_states)      # (B, n_actions)
        next_actions = next_q_online.argmax(dim=1, keepdim=True)  # (B,1)
        next_q_target = target_net(next_states)      # (B, n_actions)
        next_q_target_selected = next_q_target.gather(1, next_actions)  # (B,1)
        target_q = rewards + gamma * (1.0 - dones) * next_q_target_selected

    loss = F.mse_loss(q_values, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


### -------------------
### Training loop
### -------------------
def train_ddqn(env, baseline_map, device='cpu',
                episodes=800, steps_per_ep=500,
                replay_capacity=200000, batch_size=512,
                gamma=0.99, lr=1e-4,
                eps_start=1.0, eps_final=0.02, eps_decay_frames=200000,
                target_update_freq=1000, save_dir='runs_ddqn'):
    os.makedirs(save_dir, exist_ok=True)
    obs_dim = env.observation_spaces[env.agents[0]].shape[0]
    n_actions = env.action_spaces[env.agents[0]].n

    online_net = DuelingQNet(obs_dim, n_actions).to(device)
    target_net = DuelingQNet(obs_dim, n_actions).to(device)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = optim.Adam(online_net.parameters(), lr=lr)

    replay = ReplayBuffer(replay_capacity)

    # metrics storage
    metrics = {
        'episode': [], 'episode_reward': [], 'avg_reward_per_agent': [], 'loss': []
    }

    # epsilon schedule (linear by frames)
    frame_idx = 0
    eps = eps_start

    # bookkeeping for target update
    steps_since_target = 0

    for ep in range(episodes):
        # sample a starting hour for this episode to anchor baseline (ensures training across hours)
        start_hour = random.randint(0, 23)
        env.baseline_counts = baseline_map[start_hour]

        obs = env.reset()  # returns observation for first agent (we'll use step_all_agents)
        ep_reward_sum = 0.0
        ep_agent_rewards = {a: 0.0 for a in env.agents}
        losses = []

        for t in range(steps_per_ep):
            # collect states & actions for all agents (parameter sharing)
            actions = {}
            states_batch = {}
            # build actions with epsilon-greedy
            for a in env.agents:
                s = env.observe(a)  # np.array shape (obs_dim,)
                states_batch[a] = s
                frame_idx += 1
                # update epsilon
                eps = max(eps_final, eps_start - (eps_start - eps_final) * (frame_idx / eps_decay_frames))
                if random.random() < eps:
                    action = random.randrange(n_actions)
                else:
                    with torch.no_grad():
                        st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                        qvals = online_net(st)
                        action = int(qvals.argmax(dim=1).item())
                actions[a] = action

            # step all agents in parallel
            rewards_dict = env.step_all_agents(actions)

            # record transitions: for each agent we store (s,a,r,s',done)
            # To get next_state we observe the agent after step_all_agents
            for a in env.agents:
                s = states_batch[a]
                a_act = actions[a]
                r = float(rewards_dict[a])
                s_next = env.observe(a)
                done = bool(env.dones[a])
                replay.push(s, a_act, r, s_next, done)

                ep_reward_sum += r
                ep_agent_rewards[a] += r

            # learning step
            if len(replay) >= batch_size:
                batch = replay.sample(batch_size)
                loss_val = ddqn_update(online_net, target_net, optimizer, batch, gamma=gamma, device=device)
                losses.append(loss_val)

            # target network update
            steps_since_target += 1
            if steps_since_target >= target_update_freq:
                target_net.load_state_dict(online_net.state_dict())
                steps_since_target = 0

            # termination check (we use env.dones for all agents)
            if all(env.dones.values()):
                break

        # Episode metrics
        mean_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
        mean_episode_reward = ep_reward_sum
        mean_per_agent = np.mean(list(ep_agent_rewards.values()))

        metrics['episode'].append(ep)
        metrics['episode_reward'].append(mean_episode_reward)
        metrics['avg_reward_per_agent'].append(mean_per_agent)
        metrics['loss'].append(mean_loss)

        # Print progress
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Ep {ep+1}/{episodes} | EpReward {mean_episode_reward:.2f} | AvgAgent {mean_per_agent:.3f} | Loss {mean_loss:.6f} | Eps {eps:.4f}")

        # periodic save
        if (ep + 1) % 100 == 0:
            torch.save(online_net.state_dict(), os.path.join(save_dir, f"dueling_ddqn_ep{ep+1}.pth"))
            pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "metrics_partial.csv"), index=False)

    # final save
    torch.save(online_net.state_dict(), os.path.join(save_dir, "dueling_ddqn_final.pth"))
    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "metrics_final.csv"), index=False)

    return online_net, target_net, metrics, replay


### -------------------
### Evaluation: per-hour average density & congestion index
### -------------------
def evaluate_hourly(env, model, baseline_map, hours, eval_episodes=5, steps_per_ep=500, device='cpu'):
    """
    For each hour in `hours`, set env.baseline_counts accordingly, run several episodes (greedy),
    and compute:
      - mean_density_per_segment (averaged across timesteps and episodes)
      - congestion_index = std of mean_density_per_segment
    Stores results in a dict hour -> DataFrame(segment, mean_density) and dict for congestion index.
    """
    results_density = {}
    congestion = {}

    for hour in hours:
        env.baseline_counts = baseline_map[hour]
        # accumulate per-segment densities across episodes & timesteps
        accum = np.zeros(env.num_segments, dtype=np.float64)
        frames = 0
        for ep in range(eval_episodes):
            env.reset()
            for t in range(steps_per_ep):
                # build greedy actions for all agents
                actions = {}
                for a in env.agents:
                    s = env.observe(a)
                    st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        qvals = model(st)
                    action = int(qvals.argmax(dim=1).item())
                    actions[a] = action
                env.step_all_agents(actions)
                # accumulate densities snapshot
                accum += env.segment_density
                frames += 1
                if all(env.dones.values()):
                    break
        mean_density = accum / max(1, frames)
        # store as DataFrame
        df = pd.DataFrame({'segment': np.arange(env.num_segments), 'mean_density': mean_density})
        results_density[hour] = df
        congestion[hour] = float(np.std(mean_density))
    return results_density, congestion


### -------------------
### Main entry
### -------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # load CSVs and build hourly baseline dataframe
    print("Loading CSV traffic data ...")
    all_df = load_kaggle_traffic_csvs(args.data_dir)  # user-supplied function
    baseline_df = build_hourly_baselines(all_df)      # user-supplied function
    # baseline_df should have columns ['sensor','hour','avg_count']

    # build per-hour -> segment map
    baseline_map = build_hourly_baseline_map(baseline_df, args.num_segments)

    # create env (do not modify the environment implementation)
    env = TrafficEnv(n_agents=args.num_agents, num_segments=args.num_segments, baseline_counts=None, neighbor_radius=args.neighbor_radius)

    # train ddqn
    online_net, target_net, metrics, replay = train_ddqn(
        env,
        baseline_map,
        device=device,
        episodes=args.episodes,
        steps_per_ep=args.steps_per_episode,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        gamma=args.gamma,
        lr=args.lr,
        eps_start=args.eps_start,
        eps_final=args.eps_final,
        eps_decay_frames=args.eps_decay_frames,
        target_update_freq=args.target_update_freq,
        save_dir=args.save_dir
    )

    # evaluate for hours 0..3 (or user-specified)
    hours_to_eval = args.eval_hours if args.eval_hours is not None else [0, 1, 2, 3]
    results_density, congestion = evaluate_hourly(env, online_net, baseline_map, hours_to_eval, eval_episodes=args.eval_episodes, steps_per_ep=args.steps_per_episode, device=device)

    # save evaluation results to CSV
    eval_dir = os.path.join(args.save_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    for h, df in results_density.items():
        df.to_csv(os.path.join(eval_dir, f"mean_density_hour_{h}.csv"), index=False)
    # save congestion per hour
    congestion_df = pd.DataFrame({'hour': list(congestion.keys()), 'congestion_index': list(congestion.values())})
    congestion_df.to_csv(os.path.join(eval_dir, "congestion_index.csv"), index=False)

    print("Training and evaluation completed. Artifacts saved under:", args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./Data', help='Path to folder with CSVs')
    parser.add_argument('--num_agents', type=int, default=80)
    parser.add_argument('--num_segments', type=int, default=50)
    parser.add_argument('--neighbor_radius', type=int, default=1)
    parser.add_argument('--episodes', type=int, default=800)
    parser.add_argument('--steps_per_episode', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--replay_capacity', type=int, default=200000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_final', type=float, default=0.02)
    parser.add_argument('--eps_decay_frames', type=int, default=200000)
    parser.add_argument('--target_update_freq', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='runs_ddqn')
    parser.add_argument('--eval_episodes', type=int, default=5)
    parser.add_argument('--eval_hours', nargs='*', type=int, default=[0,1,2,3])
    args = parser.parse_args()

    main(args)
