# =============================================================
# models.py  —  SmartCity Traffic Control System
# =============================================================
# RULE: Action, Observation, State must inherit from OpenEnv
# base classes. This is a mandatory hackathon requirement.
# =============================================================

from pydantic import Field
from typing import List, Any, Dict
from openenv.core.env_server.types import (
    Action      as OpenEnvAction,
    Observation as OpenEnvObservation,
    State       as OpenEnvState,
)


# =============================================================
# ACTION — what one agent sends each step
# =============================================================

class TrafficAction(OpenEnvAction):
    """
    One agent's decision: which lane gets the green light.

    Example:
        action = TrafficAction(agent_id=0, phase=2)
        # Agent 0 turns East lane green
    """
    agent_id: int = Field(
        ...,
        description="Which intersection is acting (0, 1, 2, or 3)"
    )
    phase: int = Field(
        ...,
        ge=0,
        le=3,
        description="Which lane gets green: 0=North 1=South 2=East 3=West"
    )


# =============================================================
# OBSERVATION — what environment sends back to one agent
# =============================================================

class TrafficObservation(OpenEnvObservation):
    """
    Everything one agent can see after each step.
    The 8-number state vector your problem statement defined.

    Example:
        obs = TrafficObservation(
            agent_id=0,
            lane_counts=[10, 3, 7, 2],
            neighbor_totals=[18, 5],
            time_slot=2,
            emergency_flag=0,
            reward=-22.0,
            done=False
        )
    """
    agent_id: int = Field(
        ...,
        description="Which intersection this observation belongs to"
    )
    lane_counts: List[int] = Field(
        default=[0, 0, 0, 0],
        description="Cars in [North, South, East, West] lanes at own intersection"
    )
    neighbor_totals: List[int] = Field(
        default=[0, 0],
        description="Total cars at [left_neighbor, right_neighbor]"
    )
    time_slot: int = Field(
        default=1,
        ge=0,
        le=2,
        description="Time of day: 0=night  1=normal  2=rush hour"
    )
    emergency_flag: int = Field(
        default=0,
        ge=0,
        le=1,
        description="1 if ambulance is stuck here, 0 if not"
    )
    # Note: reward and done are inherited from OpenEnvObservation
    # We do NOT redefine them here — that would cause a conflict


# =============================================================
# STATE — full city snapshot (used internally by the server)
# =============================================================

class CityState(OpenEnvState):
    """
    Complete snapshot of all 4 intersections.
    The server tracks this; agents only see their own Observation.

    Example:
        state = CityState(
            step=45,
            all_lane_counts=[[10,3,7,2],[5,1,8,0],[3,9,2,4],[6,0,3,5]],
            current_phases=[2, 0, 1, 3],
            time_slot=2,
            emergency_flags=[0, 0, 1, 0],
            episode_reward=-450.0,
            done=False
        )
    """
    step: int = Field(
        default=0,
        ge=0,
        description="Current step (0 to 199)"
    )
    all_lane_counts: List[List[int]] = Field(
        default=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
        description="Cars in every lane at every intersection [4 agents][4 lanes]"
    )
    current_phases: List[int] = Field(
        default=[0, 0, 0, 0],
        description="Current green-light direction at each intersection"
    )
    time_slot: int = Field(
        default=1,
        ge=0,
        le=2,
        description="0=night  1=normal  2=rush hour"
    )
    emergency_flags: List[int] = Field(
        default=[0, 0, 0, 0],
        description="1 where ambulance is present, 0 otherwise"
    )
    episode_reward: float = Field(
        default=0.0,
        description="Running total reward so far this episode"
    )


# =============================================================
# QUICK TEST — run: python models.py
# =============================================================

if __name__ == "__main__":
    print("Testing models...")

    a = TrafficAction(agent_id=0, phase=2)
    print(f"Action OK:      agent={a.agent_id} phase={a.phase}")

    obs = TrafficObservation(
        agent_id=1,
        lane_counts=[10, 3, 7, 2],
        neighbor_totals=[18, 5],
        time_slot=2,
        emergency_flag=0,
        reward=-22.0,
        done=False
    )
    print(f"Observation OK: agent={obs.agent_id} lanes={obs.lane_counts}")

    s = CityState(
        step=10,
        all_lane_counts=[[10,3,7,2],[5,1,8,0],[3,9,2,4],[6,0,3,5]],
        current_phases=[2, 0, 1, 3],
        time_slot=2,
        emergency_flags=[0, 0, 1, 0],
        episode_reward=-220.0,
        done=False
    )
    print(f"CityState OK:   step={s.step} emergency at {s.emergency_flags.index(1)}")
    print("\nAll models working correctly!")
