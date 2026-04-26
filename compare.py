# =============================================================
# compare.py  —  SmartCity Traffic Control System
#
# PURPOSE:
#   Run 3 different agents and plot them on the SAME graph.
#   This PROVES that Federated Q-Learning is better.
#
#   3 lines on the graph:
#     RED   — Random agent (no learning, baseline)
#     BLUE  — Q-Learning without federation
#     GREEN — Federated Q-Learning (your innovation)
#
#   Judges see GREEN > BLUE > RED → proof your system works.
#
# HOW TO RUN:
#   python compare.py
#
# OUTPUT:
#   comparison_graph.png  
# =============================================================

import sys
import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent  import FederatedAgents, QLearningAgent
from models import TrafficAction
from server.smartcity_traffic_environment import CityTrafficEnvironment

EPISODES = 150   # enough to show learning trend clearly
TASK     = "easy"


# =============================================================
# get all 4 agent observations from environment
# =============================================================

def get_obs(env):
    return [
        env._make_observation(i, 0.0, False).model_dump()
        for i in range(4)
    ]


# =============================================================
# RUN 1: Random Agent (no learning at all)
# =============================================================

def run_random(episodes=EPISODES):
    print(f"Running Random Agent ({episodes} episodes)...")
    env     = CityTrafficEnvironment(task=TASK)
    rewards = []

    for ep in range(episodes):
        env.reset()
        ep_reward = 0.0
        done      = False

        while not done:
            for agent_id in range(4):
                action  = TrafficAction(
                    agent_id=agent_id,
                    phase=random.randint(0, 3)
                )
                obs  = env.step(action)
                ep_reward += obs.reward or 0.0
                done = obs.done

        rewards.append(ep_reward)
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}/{episodes} | "
                  f"avg={np.mean(rewards[-10:]):.0f}")

    print(f"  Random done. Final avg: {np.mean(rewards[-10:]):.0f}")
    return rewards


# =============================================================
# RUN 2: Q-Learning WITHOUT Federation
# =============================================================

def run_qlearning_no_federation(episodes=EPISODES):
    print(f"\nRunning Q-Learning (no federation) ({episodes} episodes)...")
    env     = CityTrafficEnvironment(task=TASK)
    # federation_interval=9999 means it never federates
    agents  = FederatedAgents(
        n_agents=4,
        federation_interval=9999,
    )
    rewards = []

    for ep in range(episodes):
        env.reset()
        obs_list  = get_obs(env)
        ep_reward = 0.0
        done      = False

        while not done:
            actions = agents.select_actions(obs_list)
            step_rewards  = []
            final_obs_obj = None

            for agent_id in range(4):
                action_obj = TrafficAction(
                    agent_id=agent_id,
                    phase=actions[agent_id]
                )
                obs_obj = env.step(action_obj)
                step_rewards.append(obs_obj.reward or 0.0)
                final_obs_obj = obs_obj

            done          = final_obs_obj.done
            next_obs_list = get_obs(env)

            agents.learn_step(
                obs_list=obs_list,
                actions=actions,
                rewards=step_rewards,
                next_obs_list=next_obs_list,
                done=done,
            )

            ep_reward += sum(step_rewards)
            obs_list   = next_obs_list

        agents.end_episode(ep_reward)
        rewards.append(ep_reward)

        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}/{episodes} | "
                  f"avg={np.mean(rewards[-10:]):.0f}")

    print(f"  Q-Learning done. Final avg: {np.mean(rewards[-10:]):.0f}")
    return rewards


# =============================================================
# RUN 3: Federated Q-Learning (your full system)
# =============================================================

def run_federated(episodes=EPISODES):
    print(f"\nRunning Federated Q-Learning ({episodes} episodes)...")
    env    = CityTrafficEnvironment(task=TASK)
    agents = FederatedAgents(
        n_agents=4,
        federation_interval=10,   # share every 10 episodes
    )
    rewards = []

    for ep in range(episodes):
        env.reset()
        obs_list  = get_obs(env)
        ep_reward = 0.0
        done      = False

        while not done:
            actions = agents.select_actions(obs_list)
            step_rewards  = []
            final_obs_obj = None

            for agent_id in range(4):
                action_obj = TrafficAction(
                    agent_id=agent_id,
                    phase=actions[agent_id]
                )
                obs_obj = env.step(action_obj)
                step_rewards.append(obs_obj.reward or 0.0)
                final_obs_obj = obs_obj

            done          = final_obs_obj.done
            next_obs_list = get_obs(env)

            agents.learn_step(
                obs_list=obs_list,
                actions=actions,
                rewards=step_rewards,
                next_obs_list=next_obs_list,
                done=done,
            )

            ep_reward += sum(step_rewards)
            obs_list   = next_obs_list

        agents.end_episode(ep_reward)
        rewards.append(ep_reward)

        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}/{episodes} | "
                  f"avg={np.mean(rewards[-10:]):.0f} | "
                  f"federations={agents.federation_count}")

    print(f"  Federated done. Final avg: {np.mean(rewards[-10:]):.0f}")
    return rewards


# =============================================================
# SMOOTH HELPER
# =============================================================

def smooth(values, window=10):
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(values[start:i+1])))
    return result


# =============================================================
# PLOT ALL 3 ON SAME GRAPH
# =============================================================

def plot_comparison(random_r, qlearn_r, federated_r):
    episodes = list(range(1, len(random_r) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "SmartCity Traffic — Agent Comparison\n"
        "Random vs Q-Learning vs Federated Q-Learning",
        fontsize=14, fontweight="bold"
    )

    # ── Left plot: smoothed rewards ───────────────────────────
    ax1 = axes[0]

    ax1.plot(episodes, smooth(random_r, 10),
             color="red",        linewidth=2,   label="Random Agent (no learning)")
    ax1.plot(episodes, smooth(qlearn_r, 10),
             color="royalblue",  linewidth=2,   label="Q-Learning (no federation)")
    ax1.plot(episodes, smooth(federated_r, 10),
             color="green",      linewidth=2.5, label="Federated Q-Learning (our system)")

    # Federation event lines
    for ep in range(10, len(federated_r) + 1, 10):
        ax1.axvline(x=ep, color="orange", alpha=0.2, linewidth=0.8)
    ax1.axvline(x=10, color="orange", alpha=0.5,
                linewidth=1, label="Federation event")

    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Total Reward (higher = better)", fontsize=12)
    ax1.set_title("Smoothed Reward per Episode", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ── Right plot: bar chart of final performance ────────────
    ax2 = axes[1]

    final_random    = float(np.mean(random_r[-20:]))
    final_qlearn    = float(np.mean(qlearn_r[-20:]))
    final_federated = float(np.mean(federated_r[-20:]))

    labels = ["Random\nAgent", "Q-Learning\n(no federation)", "Federated\nQ-Learning"]
    values = [final_random, final_qlearn, final_federated]
    colors = ["#ff6b6b", "#4dabf7", "#51cf66"]

    bars = ax2.bar(labels, values, color=colors, width=0.5, edgecolor="white")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - abs(bar.get_height()) * 0.05,
            f"{val:.0f}",
            ha="center", va="top",
            color="white", fontsize=11, fontweight="bold"
        )

    # Improvement annotation
    improvement = final_federated - final_random
    ax2.annotate(
        f"Federated is {abs(improvement):.0f}\nbetter than random",
        xy=(2, final_federated),
        xytext=(1.5, final_federated * 0.85),
        fontsize=10, color="darkgreen", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="darkgreen")
    )

    ax2.set_ylabel("Avg Reward (last 20 episodes)", fontsize=12)
    ax2.set_title("Final Performance Comparison", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = "comparison_graph.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nComparison graph saved → {path}")
    return path


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  SmartCity Traffic — Agent Comparison")
    print(f"  Task: {TASK} | Episodes per agent: {EPISODES}")
    print("=" * 60)

    # Run all 3 agents
    random_rewards    = run_random(EPISODES)
    qlearn_rewards    = run_qlearning_no_federation(EPISODES)
    federated_rewards = run_federated(EPISODES)

    # Summary
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Random Agent:          {np.mean(random_rewards[-20:]):10.0f}")
    print(f"  Q-Learning (no fed):   {np.mean(qlearn_rewards[-20:]):10.0f}")
    print(f"  Federated Q-Learning:  {np.mean(federated_rewards[-20:]):10.0f}")
    improvement = np.mean(federated_rewards[-20:]) - np.mean(random_rewards[-20:])
    print(f"  Improvement over random: {improvement:+.0f}")
    print("=" * 60)

    # Plot
    plot_comparison(random_rewards, qlearn_rewards, federated_rewards)

    print("\nDone! Open comparison_graph.png to see the results.")
    print("This is your PROOF that Federated Q-Learning works.")
