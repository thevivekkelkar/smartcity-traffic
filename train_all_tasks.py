# =============================================================
# train_all_tasks.py  —  SmartCity Traffic Control System
#
# Runs training on ALL 4 difficulty levels and generates
# a combined comparison graph showing improvement across
# easy, medium, hard and expert tasks.
#
# HOW TO RUN:
#   python train_all_tasks.py
#
# OUTPUT:
#   saved_agents/          — trained agents per task
#   all_tasks_curves.png   — combined reward curves (show judges!)
#   all_tasks_results.json — full results
# =============================================================

import sys, os, time, json, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent  import FederatedAgents
from models import TrafficAction
from server.smartcity_traffic_environment import CityTrafficEnvironment

# =============================================================
# CONFIGURATION
# =============================================================

TASKS = ["easy", "medium", "hard", "expert"]

CONFIG = {
    "total_episodes"      : 200,
    "learning_rate"       : 0.1,
    "discount"            : 0.95,
    "epsilon_decay"       : 0.992,
    "federation_interval" : 10,
    "print_every"         : 50,
}

TASK_COLORS = {
    "easy"  : "#2ecc71",
    "medium": "#3498db",
    "hard"  : "#e67e22",
    "expert": "#e74c3c",
}

# =============================================================
# HELPERS
# =============================================================

def get_obs(env):
    return [
        env._make_observation(i, 0.0, False).model_dump()
        for i in range(4)
    ]

def smooth(values, window=10):
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(values[start:i+1])))
    return result

# =============================================================
# TRAIN ONE TASK
# =============================================================

def train_task(task: str) -> dict:
    print(f"\n{'='*55}")
    print(f"  Training Task: {task.upper()}")
    print(f"{'='*55}")

    env    = CityTrafficEnvironment(task=task)
    agents = FederatedAgents(
        n_agents            = 4,
        federation_interval = CONFIG["federation_interval"],
        learning_rate       = CONFIG["learning_rate"],
        discount            = CONFIG["discount"],
        epsilon_decay       = CONFIG["epsilon_decay"],
    )

    episode_rewards = []
    best_reward     = float("-inf")
    start_time      = time.time()

    for episode in range(1, CONFIG["total_episodes"] + 1):
        env.reset()
        obs_list  = get_obs(env)
        ep_reward = 0.0
        done      = False

        while not done:
            actions = agents.select_actions(obs_list)
            step_rewards  = []
            final_obs_obj = None

            for agent_id in range(4):
                obs_obj = env.step(
                    TrafficAction(agent_id=agent_id, phase=actions[agent_id])
                )
                step_rewards.append(obs_obj.reward or 0.0)
                final_obs_obj = obs_obj

            done          = final_obs_obj.done
            next_obs_list = get_obs(env)

            agents.learn_step(
                obs_list      = obs_list,
                actions       = actions,
                rewards       = step_rewards,
                next_obs_list = next_obs_list,
                done          = done,
            )

            ep_reward += sum(step_rewards)
            obs_list   = next_obs_list

        agents.end_episode(ep_reward)
        episode_rewards.append(ep_reward)

        if ep_reward > best_reward:
            best_reward = ep_reward

        if episode % CONFIG["print_every"] == 0:
            avg = float(np.mean(episode_rewards[-10:]))
            print(f"  Ep {episode:4d}/200 | avg10={avg:9.1f} | "
                  f"best={best_reward:9.1f} | eps={agents.get_epsilon():.3f}")

    total_time  = time.time() - start_time
    first_avg   = float(np.mean(episode_rewards[:10]))
    last_avg    = float(np.mean(episode_rewards[-10:]))
    improvement = last_avg - first_avg

    print(f"\n  Task {task} complete!")
    print(f"  Improvement: {improvement:+.1f}")
    print(f"  Time: {total_time:.1f}s")

    # Save agents
    save_dir = f"saved_agents/{task}"
    agents.save_all(save_dir)

    return {
        "task"           : task,
        "episode_rewards": episode_rewards,
        "best_reward"    : best_reward,
        "first_avg_10"   : first_avg,
        "last_avg_10"    : last_avg,
        "improvement"    : improvement,
        "training_time_s": total_time,
        "federation_runs": agents.federation_count,
    }

# =============================================================
# PLOT ALL TASKS COMBINED
# =============================================================

def plot_all_tasks(results: list):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "SmartCity Traffic — Training Across All Difficulty Levels\n"
        "Federated Q-Learning: Easy → Medium → Hard → Expert",
        fontsize=13, fontweight="bold"
    )

    episodes = list(range(1, CONFIG["total_episodes"] + 1))

    # ── Left: smoothed reward curves ──
    ax1 = axes[0]
    for r in results:
        s = smooth(r["episode_rewards"], window=15)
        ax1.plot(
            episodes, s,
            color=TASK_COLORS[r["task"]],
            linewidth=2,
            label=f"{r['task'].capitalize()} (+{r['improvement']:.0f})"
        )

    # Federation event lines
    for ep in range(10, CONFIG["total_episodes"]+1, 10):
        ax1.axvline(x=ep, color="orange", alpha=0.15, linewidth=0.6)
    ax1.axvline(x=10, color="orange", alpha=0.4, linewidth=1,
                label="Federation event")

    ax1.set_xlabel("Episode", fontsize=11)
    ax1.set_ylabel("Total Reward (higher = better)", fontsize=11)
    ax1.set_title("Reward Improvement per Task", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ── Right: improvement bar chart ──
    ax2 = axes[1]
    task_names  = [r["task"].capitalize() for r in results]
    improvements = [r["improvement"] for r in results]
    bar_colors  = [TASK_COLORS[r["task"]] for r in results]

    bars = ax2.bar(task_names, improvements, color=bar_colors,
                   width=0.5, edgecolor="white")

    for bar, val in zip(bars, improvements):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + abs(max(improvements)) * 0.02,
            f"+{val:.0f}",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold",
            color=bar.get_facecolor()
        )

    ax2.set_xlabel("Task Difficulty", fontsize=11)
    ax2.set_ylabel("Reward Improvement (first 10 → last 10 episodes)", fontsize=11)
    ax2.set_title("Improvement by Difficulty Level", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = "all_tasks_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nCombined graph saved → {path}")
    return path


# =============================================================
# MAIN
# =============================================================

def main():
    print("=" * 55)
    print("  SmartCity Traffic — Full Training (All Tasks)")
    print("=" * 55)
    print(f"  Tasks:    {TASKS}")
    print(f"  Episodes: {CONFIG['total_episodes']} per task")
    print(f"  Total:    {len(TASKS) * CONFIG['total_episodes']} episodes")
    print("=" * 55)

    os.makedirs("saved_agents", exist_ok=True)

    all_results = []
    total_start = time.time()

    for task in TASKS:
        result = train_task(task)
        all_results.append(result)

    total_time = time.time() - total_start

    # Print summary
    print("\n" + "=" * 55)
    print("  ALL TASKS COMPLETE!")
    print("=" * 55)
    for r in all_results:
        print(f"  {r['task']:<8} | improvement={r['improvement']:+8.1f} | "
              f"best={r['best_reward']:10.1f}")
    print(f"\n  Total training time: {total_time:.1f}s")
    print("=" * 55)

    # Plot
    plot_all_tasks(all_results)

    # Save JSON
    with open("all_tasks_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  Files saved:")
    print(f"    all_tasks_curves.png  ← show this to judges!")
    print(f"    all_tasks_results.json")
    print(f"    saved_agents/easy/    ")
    print(f"    saved_agents/medium/  ")
    print(f"    saved_agents/hard/    ")
    print(f"    saved_agents/expert/  ")
    print(f"\n  Open all_tasks_curves.png to see combined results!")

    return all_results


if __name__ == "__main__":
    main()
