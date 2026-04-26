# =============================================================
# inference.py  —  SmartCity Traffic Control System
#
# THIS FILE IS MANDATORY by hackathon rules.
# It must be in the ROOT directory of the project.
#
# PURPOSE:
#   Loads a trained agent and runs ONE complete episode.
#   It also serves as a demo — run this to show the system live.
#
# HOW TO RUN:
#   Make sure server is running first:
#     Terminal 1: cd server && python app.py
#   Then run inference:
#     Terminal 2: python inference.py
#
# HOW TO RUN WITHOUT SERVER (standalone mode):
#     python inference.py --standalone
# =============================================================

import sys
import os
import random
import json
import argparse

# Add current directory so we can import our files
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================
# STANDALONE MODE — runs without a server
# Uses the environment directly (no HTTP)
# =============================================================

def run_standalone(task: str = "medium", verbose: bool = True):
    """
    Run inference without needing the server running.
    Good for quick testing on your laptop.

    Args:
        task:    difficulty — "easy", "medium", "hard", "expert"
        verbose: print step-by-step output
    """
    from server.smartcity_traffic_environment import CityTrafficEnvironment
    from models import TrafficAction
    from agent import FederatedAgents

    print("=" * 60)
    print("  SmartCity Traffic — Inference (Standalone Mode)")
    print("=" * 60)
    print(f"  Task: {task}")
    print(f"  Mode: Using trained agents if saved, else greedy")
    print("=" * 60)

    env    = CityTrafficEnvironment(task=task)
    agents = FederatedAgents(n_agents=4)

    # Try to load trained agents
    save_dir = "saved_agents"
    if os.path.exists(save_dir):
        try:
            agents.load_all(save_dir)
            print(f"  Loaded trained agents from {save_dir}/")
            # Set epsilon to 0 — pure exploitation, no exploration
            for agent in agents.agents:
                agent.epsilon = 0.0
        except Exception as e:
            print(f"  Could not load agents ({e}), using greedy fallback")
    else:
        print("  No saved agents found — using greedy (always serve busiest lane)")

    # Run one episode
    obs = env.reset()
    total_reward = 0.0
    step         = 0
    done         = False

    print(f"\n  Starting episode...")
    print(f"  {'Step':<6} {'Reward':<10} {'Total':<12} {'Time':<8} {'Emergency'}")
    print(f"  {'-'*50}")

    # Collect observations for all agents first
    obs_list = [env._make_observation(i, 0.0, False) for i in range(4)]

    while not done:
        # Each agent picks an action
        actions_list = agents.select_actions(
            [o.model_dump() for o in obs_list]
        )

        # Step all 4 agents
        step_reward = 0.0
        final_obs   = None
        for agent_id in range(4):
            action = TrafficAction(
                agent_id = agent_id,
                phase    = actions_list[agent_id]
            )
            final_obs = env.step(action)
            step_reward += final_obs.reward or 0.0

        done          = final_obs.done
        total_reward += step_reward
        step         += 1

        # Rebuild obs_list for next step
        obs_list = [env._make_observation(i, 0.0, done) for i in range(4)]

        if verbose and step % 20 == 0:
            s = env.state
            time_names = {0: "night", 1: "normal", 2: "RUSH"}
            emerg = sum(s.emergency_flags)
            print(f"  {step:<6} {step_reward:<10.1f} {total_reward:<12.1f} "
                  f"{time_names[s.time_slot]:<8} {emerg} ambulance(s)")

    print(f"\n  Episode complete!")
    print(f"  Total steps:  {step}")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Avg/step:     {total_reward/step:.2f}")
    print(f"\n  Baseline (random agent) typical: ~-8000 to -12000")
    print(f"  Trained agent typical:           ~-4000 to -7000")

    return total_reward


# =============================================================
# SERVER MODE — connects to running FastAPI server
# =============================================================

def run_server_mode(
    server_url: str  = "http://localhost:8000",
    task:       str  = "medium",
    verbose:    bool = True
):
    """
    Run inference by connecting to the running OpenEnv server.
    This is the production mode judges will use.

    Args:
        server_url: URL of the running server
        task:       difficulty level
        verbose:    print step-by-step output
    """
    import requests
    from agent import FederatedAgents

    print("=" * 60)
    print("  SmartCity Traffic — Inference (Server Mode)")
    print("=" * 60)
    print(f"  Server: {server_url}")
    print(f"  Task:   {task}")
    print("=" * 60)

    # Check server health
    try:
        r = requests.get(f"{server_url}/health", timeout=5)
        health = r.json()
        print(f"  Server status: {health.get('status', 'unknown')}")
    except Exception as e:
        print(f"  ERROR: Cannot reach server at {server_url}")
        print(f"  Make sure you ran: cd server && python app.py")
        print(f"  Error: {e}")
        return None

    # Load trained agents
    agents  = FederatedAgents(n_agents=4)
    save_dir = "saved_agents"
    if os.path.exists(save_dir):
        try:
            agents.load_all(save_dir)
            for agent in agents.agents:
                agent.epsilon = 0.0
            print(f"  Loaded trained agents from {save_dir}/")
        except Exception:
            print("  Using greedy agents (no saved agents found)")

    # Reset environment via server
    r       = requests.post(f"{server_url}/reset", json={"task": task})
    state   = r.json()
    done    = False
    total   = 0.0
    step    = 0

    print(f"\n  Running episode...\n")

    while not done:
        # Build obs from state
        all_lanes = state.get("all_lane_counts", [[0]*4]*4)
        obs_list  = []
        NEIGHBORS = {0:[1,2], 1:[0,3], 2:[0,3], 3:[1,2]}
        for i in range(4):
            obs_list.append({
                "agent_id":        i,
                "lane_counts":     all_lanes[i],
                "neighbor_totals": [sum(all_lanes[n]) for n in NEIGHBORS[i]],
                "time_slot":       state.get("time_slot", 1),
                "emergency_flag":  state.get("emergency_flags", [0]*4)[i],
                "reward":          0.0,
                "done":            False,
            })

        actions = agents.select_actions(obs_list)

        # Send step to server — one action per agent
        step_reward = 0.0
        for agent_id in range(4):
            resp = requests.post(
                f"{server_url}/step",
                json={"agent_id": agent_id, "phase": actions[agent_id]}
            )
            result    = resp.json()
            done      = result.get("done", False)
            step_reward += result.get("reward") or 0.0

        total += step_reward
        step  += 1

        # Get updated state
        state = requests.get(f"{server_url}/state").json()

        if verbose and step % 20 == 0:
            print(f"  Step {step:3d} | reward={step_reward:8.1f} | "
                  f"total={total:10.1f} | done={done}")

        if done:
            break

    print(f"\n  Episode complete! Steps={step} Total reward={total:.1f}")
    return total


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="SmartCity Traffic — Run trained agent inference"
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Run without server (direct environment access)"
    )
    parser.add_argument(
        "--task",
        default="medium",
        choices=["easy", "medium", "hard", "expert"],
        help="Difficulty level (default: medium)"
    )
    parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="Server URL for server mode"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less output"
    )
    args = parser.parse_args()

    verbose = not args.quiet

    if args.standalone:
        run_standalone(task=args.task, verbose=verbose)
    else:
        run_server_mode(
            server_url = args.server,
            task       = args.task,
            verbose    = verbose
        )
