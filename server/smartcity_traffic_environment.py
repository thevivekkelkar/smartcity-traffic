# =============================================================
# server/smartcity_traffic_environment.py
# SmartCity Traffic Control System — Round 2
#
# step() now takes a single TrafficAction (OpenEnv standard)
# state is now a @property (OpenEnv standard)
# reset() now accepts seed and episode_id (OpenEnv standard)
# =============================================================

import random
import sys
import os
from typing import Optional
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import TrafficAction, TrafficObservation, CityState

# ── OpenEnv imports ───────────────────────────────────────────
from openenv.core.env_server import Environment

# =============================================================
# GRID LAYOUT
#
#   [0] ── [1]
#    |      |
#   [2] ── [3]
#
#   0 = North-West    1 = North-East
#   2 = South-West    3 = South-East
# =============================================================

NEIGHBORS = {
    0: [1, 2],
    1: [0, 3],
    2: [0, 3],
    3: [1, 2]
}

# =============================================================
# TASK CONFIGURATIONS
# =============================================================

TASK_CONFIGS = {
    "easy": {
        "arrival_rate":     1,
        "emergency_every":  999,
        "rush_hour_active": False,
        "rush_multiplier":  1.0,
        "max_steps":        200,
    },
    "medium": {
        "arrival_rate":     2,
        "emergency_every":  50,
        "rush_hour_active": True,
        "rush_multiplier":  1.5,
        "max_steps":        200,
    },
    "hard": {
        "arrival_rate":     4,
        "emergency_every":  20,
        "rush_hour_active": True,
        "rush_multiplier":  2.0,
        "max_steps":        200,
    },
    "expert": {
        "arrival_rate":     5,
        "emergency_every":  10,
        "rush_hour_active": True,
        "rush_multiplier":  2.0,
        "max_steps":        200,
    }
}

MAX_CARS = 30
N_AGENTS = 4
N_LANES  = 4


# =============================================================
# MAIN ENVIRONMENT CLASS
# ── Inherits from OpenEnv's Environment base class ──────────
# =============================================================

class CityTrafficEnvironment(Environment):
    """
    2x2 city grid traffic environment — OpenEnv compliant.

    Differences from old version:
    - Inherits Environment (OpenEnv base class)
    - step() takes ONE TrafficAction, not a list of 4 ints
      (OpenEnv standard: one action object per call)
    - state is a @property (OpenEnv standard)
    - reset() accepts seed and episode_id (OpenEnv standard)

    How multi-agent works here:
    The TrafficAction has an agent_id field (0-3).
    The environment applies that agent's action and returns
    that agent's observation. All 4 agents call step() in turn.
    The city state updates after all 4 have acted.
    """

    def __init__(self, task: str = "easy"):
        super().__init__()              # call OpenEnv's __init__

        if task not in TASK_CONFIGS:
            raise ValueError(f"Task must be one of {list(TASK_CONFIGS.keys())}")

        self.task   = task
        self.config = TASK_CONFIGS[task]

        # Internal state — filled by reset()
        self._lane_counts     = None
        self._current_phases  = None
        self._step_count      = 0
        self._episode_reward  = 0.0
        self._emergency_flags = None
        self._time_slot       = 1
        self._episode_id      = str(uuid4())

        # Buffer: collect actions from all 4 agents before advancing
        # OpenEnv calls step() once per agent per timestep
        self._pending_actions = {}   # {agent_id: phase}
        self._step_observations = {} # {agent_id: TrafficObservation}

        self.reset()

    # ──────────────────────────────────────────────
    # RESET  (OpenEnv required signature)
    # ──────────────────────────────────────────────
    def reset(
        self,
        seed:       Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs
    ) -> TrafficObservation:
        """
        Reset environment. Returns observation for agent 0.

        OpenEnv calls reset() once at episode start.
        We reset the full city and return agent 0's initial view.

        Args:
            seed:       optional random seed for reproducibility
            episode_id: optional episode identifier string
        """
        if seed is not None:
            random.seed(seed)

        self._episode_id     = episode_id or str(uuid4())
        self._step_count     = 0
        self._episode_reward = 0.0
        self._pending_actions = {}
        self._step_observations = {}

        # Random starting cars (0-5 per lane)
        self._lane_counts = [
            [random.randint(0, 5) for _ in range(N_LANES)]
            for _ in range(N_AGENTS)
        ]

        self._current_phases  = [0] * N_AGENTS
        self._emergency_flags = [0] * N_AGENTS
        self._time_slot       = 2 if self.config["rush_hour_active"] else 1

        # Return initial observation for agent 0
        return self._make_observation(agent_id=0, reward=0.0, done=False)

    # ──────────────────────────────────────────────
    # STEP  (OpenEnv required signature)
    # ──────────────────────────────────────────────
    def step(
        self,
        action:    TrafficAction,
        timeout_s: Optional[float] = None,
        **kwargs
    ) -> TrafficObservation:
        """
        Apply one agent's action. Returns that agent's observation.

        OpenEnv calls step() once per agent. We buffer actions until
        all 4 agents have acted, then advance the simulation.

        Args:
            action: TrafficAction with agent_id (0-3) and phase (0-3)

        Returns:
            TrafficObservation for that agent
        """
        agent_id = action.agent_id
        phase    = action.phase

        # Store this agent's action
        self._pending_actions[agent_id] = phase

        # If all 4 agents have submitted actions → advance simulation
        if len(self._pending_actions) == N_AGENTS:
            self._advance_simulation()
            self._pending_actions = {}      # clear for next timestep

        # Return this agent's observation (computed in _advance_simulation)
        if agent_id in self._step_observations:
            return self._step_observations[agent_id]

        # If simulation not yet advanced (agents 0-2 waiting for agent 3)
        # return a partial observation with current state, no reward yet
        return self._make_observation(agent_id=agent_id, reward=0.0, done=False)

    # ──────────────────────────────────────────────
    # STATE  (OpenEnv required — must be @property)
    # ──────────────────────────────────────────────
    @property
    def state(self) -> CityState:
        """
        Full city state. OpenEnv calls this to inspect the environment.
        Unlike Observation (one agent's view), State shows everything.
        """
        return CityState(
            step             = self._step_count,
            all_lane_counts  = [row[:] for row in self._lane_counts],
            current_phases   = self._current_phases[:],
            time_slot        = self._time_slot,
            emergency_flags  = self._emergency_flags[:],
            episode_reward   = self._episode_reward,
            done             = self._step_count >= self.config["max_steps"]
        )

    # ──────────────────────────────────────────────
    # INTERNAL: advance city simulation one step
    # ──────────────────────────────────────────────
    def _advance_simulation(self):
        """
        Called once all 4 agents have submitted their actions.
        Updates the full city state and stores observations.
        """
        self._step_count += 1

        # Apply all actions
        for agent_id, phase in self._pending_actions.items():
            self._current_phases[agent_id] = int(phase)

        # Update city mechanics
        self._update_time_slot()
        self._update_emergencies()
        self._discharge_green_lanes()
        self._flow_between_intersections()
        self._add_arriving_cars()
        self._clamp_lanes()

        # Calculate rewards
        rewards = self._calculate_rewards()
        self._episode_reward += sum(rewards)

        done = self._step_count >= self.config["max_steps"]

        # Store observations for all agents
        self._step_observations = {
            i: self._make_observation(agent_id=i, reward=rewards[i], done=done)
            for i in range(N_AGENTS)
        }

    # ──────────────────────────────────────────────
    # INTERNAL: build one agent's observation
    # ──────────────────────────────────────────────
    def _make_observation(
        self,
        agent_id: int,
        reward:   float,
        done:     bool
    ) -> TrafficObservation:
        neighbor_totals = [
            sum(self._lane_counts[n])
            for n in NEIGHBORS[agent_id]
        ]
        return TrafficObservation(
            agent_id        = agent_id,
            lane_counts     = self._lane_counts[agent_id][:],
            neighbor_totals = neighbor_totals,
            time_slot       = self._time_slot,
            emergency_flag  = self._emergency_flags[agent_id],
            reward          = reward,
            done            = done
        )

    # ──────────────────────────────────────────────
    # INTERNAL: time slot (night/normal/rush)
    # ──────────────────────────────────────────────
    def _update_time_slot(self):
        if not self.config["rush_hour_active"]:
            self._time_slot = 1
            return
        step = self._step_count % 200
        if step < 40:
            self._time_slot = 0      # night
        elif step < 120:
            self._time_slot = 1      # normal
        elif step < 160:
            self._time_slot = 2      # rush hour
        else:
            self._time_slot = 1      # back to normal

    # ──────────────────────────────────────────────
    # INTERNAL: emergency vehicles
    # ──────────────────────────────────────────────
    def _update_emergencies(self):
        every = self.config["emergency_every"]
        for i in range(N_AGENTS):
            if self._emergency_flags[i] == 1:
                if random.random() < 0.2:        # 20% chance to clear
                    self._emergency_flags[i] = 0
        if self._step_count % every == 0 and every < 999:
            victim = random.randint(0, N_AGENTS - 1)
            self._emergency_flags[victim] = 1

    # ──────────────────────────────────────────────
    # INTERNAL: cars leave green lanes
    # ──────────────────────────────────────────────
    def _discharge_green_lanes(self):
        for i in range(N_AGENTS):
            lane = self._current_phases[i]
            out  = random.randint(2, 5)
            self._lane_counts[i][lane] = max(
                0, self._lane_counts[i][lane] - out
            )

    # ──────────────────────────────────────────────
    # INTERNAL: cars flow between intersections
    # ──────────────────────────────────────────────
    def _flow_between_intersections(self):
        """
        This is what makes us genuinely multi-agent.
        Cars leaving intersection A enter intersection B.
        Without this, agents would be independent — not cooperative.
        """
        delta = [[0] * N_LANES for _ in range(N_AGENTS)]
        for i in range(N_AGENTS):
            for neighbor in NEIGHBORS[i]:
                flow         = random.randint(1, 2)
                incoming     = random.randint(0, N_LANES - 1)
                delta[neighbor][incoming] += flow
        for i in range(N_AGENTS):
            for j in range(N_LANES):
                self._lane_counts[i][j] += delta[i][j]

    # ──────────────────────────────────────────────
    # INTERNAL: new cars arrive from outside city
    # ──────────────────────────────────────────────
    def _add_arriving_cars(self):
        base = self.config["arrival_rate"]
        if self._time_slot == 2:
            rate = int(base * self.config["rush_multiplier"])
        elif self._time_slot == 0:
            rate = max(1, base // 2)
        else:
            rate = base
        for i in range(N_AGENTS):
            for j in range(N_LANES):
                self._lane_counts[i][j] += random.randint(0, rate)

    # ──────────────────────────────────────────────
    # INTERNAL: cap all lanes at MAX_CARS
    # ──────────────────────────────────────────────
    def _clamp_lanes(self):
        for i in range(N_AGENTS):
            for j in range(N_LANES):
                self._lane_counts[i][j] = min(
                    self._lane_counts[i][j], MAX_CARS
                )

    # ──────────────────────────────────────────────
    # INTERNAL: reward formula (from problem statement)
    # ──────────────────────────────────────────────
    def _calculate_rewards(self) -> list:
        """
        reward = -sum(own_cars)
               - 0.3 × sum(neighbor_cars)
               - 50   if ambulance at red
               × 2.0  if rush hour
        """
        rewards = []
        for i in range(N_AGENTS):
            own_cars      = sum(self._lane_counts[i])
            neighbor_cars = sum(
                sum(self._lane_counts[n]) for n in NEIGHBORS[i]
            )
            emergency_penalty = 0
            if self._emergency_flags[i] == 1:
                if self._current_phases[i] != 0:   # not giving North green
                    emergency_penalty = 50

            r = (-own_cars
                 - 0.3 * neighbor_cars
                 - emergency_penalty)

            if self._time_slot == 2:
                r *= self.config["rush_multiplier"]

            rewards.append(r)
        return rewards


# =============================================================
# QUICK TEST — python server/smartcity_traffic_environment.py
# =============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("Testing CityTrafficEnvironment (OpenEnv compliant)")
    print("=" * 50)

    env = CityTrafficEnvironment(task="medium")
    obs = env.reset()

    print(f"\nInitial observation (agent 0):")
    print(f"  Lanes:     {obs.lane_counts}")
    print(f"  Neighbors: {obs.neighbor_totals}")
    print(f"  Time slot: {obs.time_slot}")

    print(f"\nFull city state:")
    s = env.state
    print(f"  All lanes: {s.all_lane_counts}")
    print(f"  Emergency: {s.emergency_flags}")

    print(f"\nRunning 5 full steps (4 agent actions each)...")
    for step in range(5):
        total_reward = 0.0
        for agent_id in range(4):
            action = TrafficAction(
                agent_id = agent_id,
                phase    = random.randint(0, 3)
            )
            obs = env.step(action)
            total_reward += obs.reward or 0.0

        print(f"  Step {step+1}: total_reward={total_reward:.1f}  "
              f"done={obs.done}  time={obs.time_slot}")

    print("\nEnvironment working correctly!")
