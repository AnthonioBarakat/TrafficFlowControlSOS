import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium import spaces


class TrafficEnv(AECEnv):
    metadata = {"render.modes": ["human"], "name": "traffic_env"}

    def __init__(self, n_agents=80, num_segments=50, max_speed=5.0, dt=1.0, baseline_counts=None, neighbor_radius=1):
        super().__init__()
        # Allow n_agents > num_segments
        self.n_agents = int(n_agents)
        self.num_segments = int(num_segments)
        self.max_speed = float(max_speed)
        self.dt = float(dt)
        self.neighbor_radius = int(neighbor_radius)
        self.baseline_counts = baseline_counts  # dict mapping segment idx -> baseline density

        self.agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.pos = np.zeros(self.n_agents, dtype=np.float32)
        self.speed = np.zeros(self.n_agents, dtype=np.float32)

        # Action and observation spaces
        self.action_spaces = {a: spaces.Discrete(3) for a in self.agents}
        # observation: [own_speed, dist_to_next, local_baseline, left_density, right_density]
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

        # Initialize
        self.reset()

    def reset(self, seed=None, return_info=False, options=None):
        """Initialize positions (even spacing + jitter) and speeds to avoid initial collisions."""
        # Even base spacing to avoid clustering collisions at start
        base_positions = np.linspace(0, self.num_segments, num=self.n_agents, endpoint=False)
        # jitter relative to average spacing (small)
        avg_spacing = max(1.0, self.num_segments / max(1.0, self.n_agents))
        jitter = np.random.uniform(-0.05 * avg_spacing, 0.05 * avg_spacing, size=self.n_agents)
        self.pos = (base_positions + jitter).astype(np.float32) % self.num_segments

        # safe initial speeds
        self.speed = np.random.uniform(self.max_speed * 0.4, self.max_speed * 0.6, size=self.n_agents).astype(np.float32)

        # reset AEC selector
        self._agent_selector = agent_selector(self.agents)
        self._agent_selector.reset()
        self.current_agent = self._agent_selector.next()

        # bookkeeping
        self.rewards = {a: 0.0 for a in self.agents}
        self.dones = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._cumulative_steps = 0
        self._steps_since_reset = 0

        # initial densities (may blend with baseline_counts if provided)
        self._update_segment_density()

        # Return initial observation for the first agent to keep compatibility
        return self._observe(self.current_agent)

    def update_hour(self, hour, baseline_map):
        """Change baseline_counts to the given hour map and update densities accordingly."""
        if baseline_map is None:
            return
        if hour in baseline_map:
            self.baseline_counts = baseline_map[hour]
        # recompute segment density to reflect baseline change
        self._update_segment_density()

    def _update_segment_density(self):
        self.segment_density = np.zeros(self.num_segments, dtype=np.float32)
        for i in range(self.n_agents):
            seg = int(self.pos[i]) % self.num_segments
            self.segment_density[seg] += 1.0
        if self.baseline_counts is not None:
            for seg, val in self.baseline_counts.items():
                if 0 <= seg < self.num_segments:
                    # blend baseline with simulated occupancy to keep realism
                    self.segment_density[seg] = 0.5 * self.segment_density[seg] + 0.5 * val

    def _observe(self, agent):
        idx = int(agent.split('_')[1])
        own_speed = float(self.speed[idx])
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
        return obs, {}, False, False

    def observe(self, agent):
        obs, _, _, _ = self._observe(agent)
        return obs

    def step(self, action):
        """AEC-style single-agent step (keeps compatibility)."""
        agent = self.current_agent
        idx = int(agent.split('_')[1])

        if self.dones[agent]:
            # advance selection
            self.current_agent = self._agent_selector.next()
            return

        # dynamics
        accel = float(action - 1)  # -1,0,+1
        self.speed[idx] = np.clip(self.speed[idx] + accel * 0.5, 0.0, self.max_speed)
        self.pos[idx] = (self.pos[idx] + self.speed[idx] * self.dt) % self.num_segments

        # update densities
        self._update_segment_density()

        # local metrics
        pos_i = self.pos[idx]
        # relative positions centered in [-num_segments/2, +num_segments/2)
        rel = (self.pos - pos_i + self.num_segments / 2.0) % self.num_segments - self.num_segments / 2.0
        local_count = float(np.sum(np.abs(rel) < 1.0))  # cars within +/-1 segment
        density_norm = min(1.0, local_count / max(1.0, self.n_agents))

        diffs = (self.pos - pos_i) % self.num_segments
        diffs[idx] = np.inf
        dist_to_next = float(np.min(diffs))

        # collision logic: ignore collisions for a few steps right after reset
        collision_happened = 0.0
        collision_penalty = 0.0
        if self._steps_since_reset > 5:
            if dist_to_next < 0.2:
                collision_happened = 1.0
                collision_penalty = -4.0 * (1.0 - dist_to_next / 0.2)  # smooth strong penalty
            elif dist_to_next < 0.5:
                # incentive to keep margin (soft penalty)
                collision_penalty = -1.0 * (0.5 - dist_to_next) / 0.5

        # safety-masked speed reward (only reward movement when reasonably safe)
        safe_speed = self.speed[idx] if dist_to_next > 0.5 else 0.0

        reward = (
            1.0 * (safe_speed / max(1e-6, self.max_speed))
            - 0.25 * density_norm
            - 0.02 * abs(accel)
            + collision_penalty
        )
        reward = float(np.clip(reward, -3.0, 3.0))

        self.rewards[agent] = reward

        # termination & infos
        self._cumulative_steps += 1
        self._steps_since_reset += 1
        done = (self._cumulative_steps >= 1000)
        self.dones[agent] = done
        self.infos[agent] = {'local_density': float(local_count), 'dist_to_next': float(dist_to_next), 'density_norm': float(density_norm)}

        # advance agent selection
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
        """Parallel step: apply actions for all agents, update densities, and return rewards dict."""
        for a, act in actions_dict.items():
            idx = int(a.split('_')[1])
            accel = float(act - 1)
            self.speed[idx] = np.clip(self.speed[idx] + accel * 0.5, 0.0, self.max_speed)
            self.pos[idx] = (self.pos[idx] + self.speed[idx] * self.dt) % self.num_segments

        self._update_segment_density()

        rewards = {}
        # compute per-agent rewards same logic as step()
        for i, a in enumerate(self.agents):
            pos_i = self.pos[i]
            rel = (self.pos - pos_i + self.num_segments / 2.0) % self.num_segments - self.num_segments / 2.0
            local_count = float(np.sum(np.abs(rel) < 1.0))
            density_norm = min(1.0, local_count / max(1.0, self.n_agents))

            diffs = (self.pos - pos_i) % self.num_segments
            diffs[i] = np.inf
            dist_to_next = float(np.min(diffs))

            collision_happened = 0.0
            collision_penalty = 0.0
            if self._steps_since_reset > 5:
                if dist_to_next < 0.2:
                    collision_happened = 1.0
                    collision_penalty = -4.0 * (1.0 - dist_to_next / 0.2)
                elif dist_to_next < 0.5:
                    collision_penalty = -1.0 * (0.5 - dist_to_next) / 0.5

            safe_speed = self.speed[i] if dist_to_next > 0.5 else 0.0
            r = (
                1.0 * (safe_speed / max(1e-6, self.max_speed))
                - 0.25 * density_norm
                + collision_penalty
            )
            r = float(np.clip(r, -3.0, 3.0))

            rewards[a] = r
            self.rewards[a] = r
            self.dones[a] = False
            self.infos[a] = {'local_density': float(local_count), 'dist_to_next': float(dist_to_next), 'density_norm': float(density_norm)}

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
