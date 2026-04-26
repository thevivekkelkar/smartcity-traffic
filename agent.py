# =============================================================
# agent.py  —  SmartCity Traffic Control System
# =============================================================
# Contains TWO classes:
#   1. QLearningAgent  — one agent, one intersection, one Q-table
#   2. FederatedAgents — manages all 4 agents + sharing logic
# =============================================================

import numpy as np
import random
import json
import os
from typing import List


# =============================================================
# CLASS 1: Single Q-Learning Agent
# =============================================================

class QLearningAgent:
    """
    One agent controlling one intersection.

    Learns by updating a Q-table every step:
        Q(state, action) = Q(state, action)
                         + lr × (reward + gamma × max_Q(next_state) - Q(state, action))

    In plain words:
        Update our guess based on what actually happened.
    """

    def __init__(
        self,
        agent_id:     int,
        n_actions:    int   = 4,      # N / S / E / W green
        learning_rate: float = 0.1,   # how fast to learn
        discount:      float = 0.95,  # how much future rewards matter
        epsilon:       float = 1.0,   # 1.0 = 100% random at start
        epsilon_min:   float = 0.05,  # never go below 5% random
        epsilon_decay: float = 0.995, # multiply epsilon by this each episode
    ):
        self.agent_id      = agent_id
        self.n_actions     = n_actions
        self.lr            = learning_rate
        self.gamma         = discount
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: dict mapping state_tuple → array of 4 Q-values
        # We use a dict (not array) because state space is huge.
        # We only store states the agent has actually visited.
        self.q_table = {}

        self.episode_count   = 0
        self.episode_rewards = []

    # ── Encode observation into a state tuple ──────────────────
    def encode_state(self, obs: dict) -> tuple:
        """
        Convert observation dict into a small tuple for Q-table key.

        We "bin" car counts into 5 categories:
            0-5   → 0 (empty)
            6-10  → 1 (light)
            11-18 → 2 (moderate)
            19-25 → 3 (heavy)
            26+   → 4 (jammed)

        This shrinks the infinite state space into manageable size.
        Result: tuple of 8 small integers.
        """
        def bin_cars(n):
            if n <= 5:  return 0
            if n <= 10: return 1
            if n <= 18: return 2
            if n <= 25: return 3
            return 4

        lanes     = obs.get("lane_counts",     [0, 0, 0, 0])
        neighbors = obs.get("neighbor_totals", [0, 0])
        time_slot = obs.get("time_slot",       1)
        emergency = obs.get("emergency_flag",  0)

        return (
            bin_cars(lanes[0]),      # North lane
            bin_cars(lanes[1]),      # South lane
            bin_cars(lanes[2]),      # East  lane
            bin_cars(lanes[3]),      # West  lane
            bin_cars(neighbors[0]),  # left neighbor total
            bin_cars(neighbors[1]),  # right neighbor total
            int(time_slot),          # 0, 1, or 2
            int(emergency),          # 0 or 1
        )

    # ── Look up Q-values for a state ───────────────────────────
    def get_q_values(self, state: tuple) -> np.ndarray:
        """Return Q-values for state. If unseen, return zeros."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    # ── Choose action ──────────────────────────────────────────
    def select_action(self, obs: dict) -> int:
        """
        Epsilon-greedy action selection.

        - Probability epsilon  → random action  (EXPLORE)
        - Probability 1-epsilon → best action   (EXPLOIT)

        Special rule: if ambulance present → always give North green.
        This is domain knowledge that helps a lot in medium/hard tasks.
        """
        if obs.get("emergency_flag", 0) == 1:
            return 0  # North green — ambulance priority

        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            state    = self.encode_state(obs)
            q_values = self.get_q_values(state)
            return int(np.argmax(q_values))

    # ── Learn from one step ────────────────────────────────────
    def learn(
        self,
        obs:      dict,
        action:   int,
        reward:   float,
        next_obs: dict,
        done:     bool,
    ):
        """
        Update Q-table using the Bellman equation.

        Q(s, a) += lr × [reward + gamma × max Q(s') − Q(s, a)]
        """
        s  = self.encode_state(obs)
        s_ = self.encode_state(next_obs)

        q_current = self.get_q_values(s)[action]

        if done:
            # No future reward after episode ends
            q_target = reward
        else:
            q_target = reward + self.gamma * np.max(self.get_q_values(s_))

        # Update only the Q-value for the action we took
        self.q_table[s][action] += self.lr * (q_target - q_current)

    # ── Decay epsilon after each episode ──────────────────────
    def decay_epsilon(self):
        """Reduce exploration rate each episode until minimum."""
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    # ── Save / Load Q-table ────────────────────────────────────
    def save(self, path: str):
        """Save Q-table to JSON file."""
        # Convert tuple keys to strings for JSON compatibility
        serializable = {
            str(k): v.tolist()
            for k, v in self.q_table.items()
        }
        data = {
            "agent_id": self.agent_id,
            "epsilon":  self.epsilon,
            "q_table":  serializable,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        """Load Q-table from JSON file."""
        if not os.path.exists(path):
            print(f"  No saved agent at {path} — starting fresh")
            return
        with open(path, "r") as f:
            data = json.load(f)
        self.epsilon = data.get("epsilon", self.epsilon_min)
        # Convert string keys back to tuples
        self.q_table = {
            tuple(map(int, k.strip("()").split(", "))): np.array(v)
            for k, v in data["q_table"].items()
        }
        print(f"  Loaded agent {self.agent_id}: "
              f"{len(self.q_table)} states, epsilon={self.epsilon:.3f}")


# =============================================================
# CLASS 2: Federated Agents (manages all 4 together)
# =============================================================

class FederatedAgents:
    """
    Manages 4 QLearningAgents with federated Q-table averaging.

    Grid layout and neighbor connections:
        [0] ─ [1]
         |     |
        [2] ─ [3]

    Neighbors:
        0 → [1, 2]    (right and below)
        1 → [0, 3]    (left  and below)
        2 → [0, 3]    (above and right)
        3 → [1, 2]    (above and left)
    """

    NEIGHBORS = {
        0: [1, 2],
        1: [0, 3],
        2: [0, 3],
        3: [1, 2],
    }

    def __init__(
        self,
        n_agents:            int   = 4,
        federation_interval: int   = 10,
        learning_rate:       float = 0.1,
        discount:            float = 0.95,
        epsilon_decay:       float = 0.995,
    ):
        self.n_agents            = n_agents
        self.federation_interval = federation_interval
        self.federation_count    = 0

        # Create one agent per intersection
        self.agents = [
            QLearningAgent(
                agent_id      = i,
                learning_rate = learning_rate,
                discount      = discount,
                epsilon_decay = epsilon_decay,
            )
            for i in range(n_agents)
        ]

        print(f"FederatedAgents ready: {n_agents} agents, "
              f"federation every {federation_interval} episodes")

    # ── Select actions for all 4 agents ───────────────────────
    def select_actions(self, obs_list: list) -> List[int]:
        """
        Each agent picks an action from its own observation.

        Args:
            obs_list: list of 4 observation dicts

        Returns:
            list of 4 ints [action_0, action_1, action_2, action_3]
        """
        return [
            self.agents[i].select_action(obs_list[i])
            for i in range(self.n_agents)
        ]

    # ── All agents learn from one step ────────────────────────
    def learn_step(
        self,
        obs_list:      list,
        actions:       List[int],
        rewards:       list,
        next_obs_list: list,
        done:          bool,
    ):
        """
        Update all 4 agents' Q-tables from one simulation step.

        Args:
            obs_list:      4 observations BEFORE the step
            actions:       4 actions taken
            rewards:       4 individual rewards received
            next_obs_list: 4 observations AFTER the step
            done:          True if episode ended
        """
        for i in range(self.n_agents):
            self.agents[i].learn(
                obs      = obs_list[i],
                action   = actions[i],
                reward   = rewards[i],
                next_obs = next_obs_list[i],
                done     = done,
            )

    # ── End of episode processing ──────────────────────────────
    def end_episode(self, episode_reward: float):
        """
        Call at end of every episode.
        Decays epsilon and triggers federation if needed.

        Args:
            episode_reward: total reward for the episode
        """
        for agent in self.agents:
            agent.episode_rewards.append(episode_reward / self.n_agents)
            agent.episode_count += 1
            agent.decay_epsilon()

        ep = self.agents[0].episode_count
        if ep % self.federation_interval == 0:
            self._federate()

    # ── Federated Q-table averaging ───────────────────────────
    def _federate(self):
        """
        Average Q-tables between neighboring agents.

        For each agent i:
            For each state seen by i OR its neighbors:
                new_Q[i][state] = mean of all who have seen that state

        This is the core innovation.
        Knowledge spreads across the city every N episodes.
        """
        self.federation_count += 1

        # Snapshot all tables BEFORE modifying anything
        snapshots = {
            i: dict(agent.q_table)
            for i, agent in enumerate(self.agents)
        }

        for i, agent in enumerate(self.agents):
            neighbor_ids = self.NEIGHBORS[i]

            # All states seen by this agent or any neighbor
            all_states = set(snapshots[i].keys())
            for n in neighbor_ids:
                all_states.update(snapshots[n].keys())

            new_table = {}
            for state in all_states:
                tables_for_state = []

                if state in snapshots[i]:
                    tables_for_state.append(snapshots[i][state])

                for n in neighbor_ids:
                    if state in snapshots[n]:
                        tables_for_state.append(snapshots[n][state])

                if tables_for_state:
                    new_table[state] = np.mean(tables_for_state, axis=0)

            agent.q_table = new_table

        ep = self.agents[0].episode_count
        print(f"  [Federation #{self.federation_count}] "
              f"Episode {ep}: Q-tables shared with neighbors ✓")

    # ── Save / Load all agents ─────────────────────────────────
    def save_all(self, directory: str = "saved_agents"):
        """Save all 4 agents to JSON files."""
        os.makedirs(directory, exist_ok=True)
        for agent in self.agents:
            path = os.path.join(directory, f"agent_{agent.agent_id}.json")
            agent.save(path)
        print(f"  Agents saved → {directory}/")

    def load_all(self, directory: str = "saved_agents"):
        """Load all 4 agents from JSON files."""
        for agent in self.agents:
            path = os.path.join(directory, f"agent_{agent.agent_id}.json")
            agent.load(path)

    # ── Utility helpers ────────────────────────────────────────
    def get_epsilon(self) -> float:
        """Current epsilon value (same for all agents)."""
        return self.agents[0].epsilon

    def get_q_table_sizes(self) -> List[int]:
        """How many unique states each agent has learned."""
        return [len(a.q_table) for a in self.agents]


# =============================================================
# QUICK TEST — run: python agent.py
# =============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("Testing FederatedAgents")
    print("=" * 50)

    agents = FederatedAgents(
        n_agents            = 4,
        federation_interval = 3,  # federate every 3 for quick test
    )

    def fake_obs(agent_id):
        return {
            "agent_id":        agent_id,
            "lane_counts":     [random.randint(0, 20) for _ in range(4)],
            "neighbor_totals": [random.randint(0, 30), random.randint(0, 30)],
            "time_slot":       random.randint(0, 2),
            "emergency_flag":  random.randint(0, 1),
            "reward":          0.0,
            "done":            False,
        }

    print("\nSimulating 15 short episodes...")

    for episode in range(1, 16):
        ep_reward = 0.0
        obs_list  = [fake_obs(i) for i in range(4)]

        for step in range(10):
            actions       = agents.select_actions(obs_list)
            next_obs_list = [fake_obs(i) for i in range(4)]
            rewards       = [random.uniform(-50, -5) for _ in range(4)]
            done          = (step == 9)

            agents.learn_step(obs_list, actions, rewards, next_obs_list, done)
            ep_reward += sum(rewards)
            obs_list   = next_obs_list

        agents.end_episode(ep_reward)
        print(f"  Episode {episode:2d} | reward={ep_reward:7.1f} | "
              f"epsilon={agents.get_epsilon():.3f} | "
              f"Q-sizes={agents.get_q_table_sizes()}")

    print(f"\nFederation happened {agents.federation_count} times")
    print("\nagent.py working correctly! ✓")
