---
title: SmartCity Traffic Control
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# SmartCity Traffic Control System

> Multi-agent RL environment — 4 AI agents manage a connected 2x2 city grid using Federated Q-Learning + HF TRL GRPO.

## Quick Links
| Resource | Link |
|---|---|
| HuggingFace Space | https://huggingface.co/spaces/Vivekkelkar/smartcity-traffic |
| Training Notebook (Qwen2.5-0.5B) | [SmartCity_TRL_Training_Qwen.ipynb](SmartCity_TRL_Training_Qwen.ipynb) |
| Blog Post | https://huggingface.co/spaces/Vivekkelkar/smartcity-traffic/blob/main/BLOG.md |

## Problem
Indian cities lose crores daily to traffic gridlock. Fixed timers don't adapt — one intersection's jam cascades to neighbors. No coordination happens.

## Environment
4 AI agents on a 2x2 city grid. Cars physically flow between intersections creating genuine multi-agent dependency.
```
[Agent 0] <-> [Agent 1]
    |               |
[Agent 2] <-> [Agent 3]
```

**State (8 numbers per agent):**
[N_cars, S_cars, E_cars, W_cars, neighbor_left, neighbor_right, time_slot, emergency_flag]

**Actions:** 0=North green, 1=South, 2=East, 3=West

**Reward formula:**
reward = -own_cars - 0.3 × neighbor_cars - 50 × emergency × 2.0_if_rush_hour

**4 difficulty levels:** easy, medium, hard, expert

## Results

### Reward Improvement — +7000 to +10000 over 200 training episodes (stochastic environment)
![Reward Curve](reward_curve.png)
*Reward improving from -113000 to -104000. Orange lines = Federated Q-Learning events every 10 episodes.*

### Agent Comparison — Federated beats Random by 6534
![Comparison](comparison_graph.png)
*Red=Random, Blue=Q-Learning without federation, Green=Federated Q-Learning (our system)*


### All 4 Difficulty Levels — Combined Training
![All Tasks](all_tasks_curves.png)
*Green=Easy(+10095), Blue=Medium(+10747), Orange=Hard(+2449), Red=Expert(+756)*

## Blog / Writeup
[Read our full writeup](BLOG.md)

## LLM Training (HF TRL GRPO)
Trained Qwen2.5-0.5B on our environment using HF TRL GRPO. The LLM reads traffic state as natural language and learns optimal signal decisions. Achieved 75% accuracy on held-out test scenarios after 3 epochs on GPU.

## Training Notebook
- [Open in Google Colab](https://colab.research.google.com/drive/1A6Rg8eA0fxktrP87bOWn3LUu8KibsvBD?usp=sharing)
- [View on GitHub](SmartCity_TRL_Training_Qwen.ipynb)
  
## Try the Environment
**Live API docs:** https://vivekkelkar-smartcity-traffic.hf.space/docs

## Innovation — Federated Q-Learning
Every 10 episodes, agents share Q-tables with neighbors. Rush-hour strategies discovered at one intersection propagate city-wide automatically. This is emergent cooperation — not programmed, but learned.

## Theme
Multi-Agent Interactions (Theme 1) — Halluminate sub-theme (Multi-Actor Environments) 
