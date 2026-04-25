---
title: SmartCity Traffic Control
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# SmartCity Traffic Control System

Multi-agent traffic control using Federated Q-Learning + HF TRL.
4 AI agents manage a 2x2 city grid — cars flow between intersections,
agents share knowledge every 10 episodes.

## Links
- **HuggingFace Space:** https://huggingface.co/spaces/Vivekkelkar/smartcity-traffic
- **Training Notebook:** https://github.com/thevivekkelkar/smartcity-traffic/blob/main/SmartCity_TRL_Training.ipynb
- **Blog Post:** (add after writing)

## Environment
- 4 intersections in a 2x2 grid
- 8-number state vector per agent
- Reward: -own_cars - 0.3*neighbor_cars - 50*emergency * 2.0 rush_hour
- Federated Q-Learning: agents share Q-tables every 10 episodes

## Results

### Reward Curve — Training Improvement (+8711)
![Reward Curve](reward_curve.png)

### Agent Comparison — Random vs Q-Learning vs Federated
![Comparison](comparison_graph.png)

## How to Run
```bash
# Reset environment
POST https://vivekkelkar-smartcity-traffic.hf.space/reset

# Take a step
POST https://vivekkelkar-smartcity-traffic.hf.space/step

# See full API
GET https://vivekkelkar-smartcity-traffic.hf.space/docs
```

## Training
See `SmartCity_TRL_Training.ipynb` — trains GPT-2 on this environment using HF TRL GRPO.
LLM achieved 100% accuracy on test scenarios.

## Theme
Multi-Agent Interactions (Theme 1) — Halluminate sub-theme
