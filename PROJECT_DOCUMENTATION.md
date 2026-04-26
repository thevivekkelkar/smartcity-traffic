# SmartCity Traffic Control System
## Complete Project Documentation
### Meta PyTorch OpenEnv Hackathon × Scaler School of Technology

**Team:** Vivek Kailas Kelkar, Rushikesh Arun Jorvekar, Sanika Patil
**Theme:** Multi-Agent Interactions (Theme 1)
**Sub-theme:** Halluminate — Multi-Actor Environments

---

# TABLE OF CONTENTS

1. What Problem Are We Solving?
2. What Is Our Solution? (Simple Version)
3. How the City Grid Works
4. The State Vector — What Each Agent Sees
5. Actions — What Each Agent Can Do
6. The Reward Formula — Why It Is Designed This Way
7. Federated Q-Learning — Our Core Innovation
8. What Is Q-Learning? (From Scratch)
9. File-by-File Explanation
10. How OpenEnv Framework Works
11. HF TRL — How We Train an LLM
12. Key Technical Terms Explained Simply

---

# 1. WHAT PROBLEM ARE WE SOLVING?

## The Real-World Problem

Every major Indian city — Mumbai, Bangalore, Delhi — loses crores of rupees every day because of traffic jams. The main reason is simple: **traffic lights run on fixed timers**.

A fixed timer gives green to North lane for 30 seconds, then South for 30 seconds, then East for 30 seconds, then West for 30 seconds — regardless of whether there are 1 car or 30 cars waiting.

**The problem gets worse** because intersections are connected. If intersection A has a jam, cars back up into intersection B, which backs up into C. One bad intersection ruins the whole city. But each intersection's traffic light does not know what is happening at neighboring intersections.

## Why Existing Solutions Fail

- **Fixed timers:** Completely blind to traffic. No adaptation.
- **Simple sensors:** Each intersection responds to its own traffic but ignores neighbors.
- **Centralized AI:** One computer controls everything — single point of failure, does not scale.

## Our Solution

4 independent AI agents, one per intersection, that:
1. See their own traffic AND their neighbors' traffic
2. Learn from experience which action reduces congestion
3. Share their learned strategies with neighboring agents every 10 episodes
4. Handle rush hours and ambulance emergencies automatically

This is not a fixed timer. This is not centralized control. This is **distributed cooperative intelligence**.

---

# 2. WHAT IS OUR SOLUTION? (SIMPLE VERSION)

Imagine 4 security guards, one at each intersection of a 2x2 city grid.

Each guard:
- Watches how many cars are waiting in each direction
- Decides which direction gets the green light
- Talks to neighboring guards every 10 minutes to share what they learned
- If they hear an ambulance — immediately gives it green

Over time, the guards get smarter. They learn that during rush hour (5-7pm), East-West lanes need more green time. They learn that when the neighboring intersection is jammed, they should hold cars a little longer to avoid making it worse.

This is exactly what our AI agents do — except they learn automatically from rewards, not from a human trainer.

---

# 3. HOW THE CITY GRID WORKS

## The 2×2 Grid Layout

```
    [Agent 0] ←——→ [Agent 1]
    NW Corner       NE Corner
         ↕               ↕
    [Agent 2] ←——→ [Agent 3]
    SW Corner       SE Corner
```

- **4 intersections** connected in a 2×2 grid
- **4 lanes per intersection:** North, South, East, West
- **16 lanes total** in the city
- **Max 30 cars per lane** before it is considered jammed

## How Cars Move

This is the most important part — what makes us genuinely multi-agent:

**Step 1:** Each intersection gives green to one lane → cars leave that lane (2-5 cars per step)

**Step 2:** Some of those cars travel to a neighboring intersection → they enter that neighbor's queue

**Step 3:** New cars arrive from outside the city → they join random lanes

**Why this matters:** Agent 0's decision directly affects how many cars Agent 1 has to deal with. They are NOT independent. They are physically connected. This is genuine multi-agent interaction.

## Neighbor Connections

```
Agent 0 neighbors: Agent 1 (right) and Agent 2 (below)
Agent 1 neighbors: Agent 0 (left)  and Agent 3 (below)
Agent 2 neighbors: Agent 0 (above) and Agent 3 (right)
Agent 3 neighbors: Agent 1 (above) and Agent 2 (left)
```

## Time Slots (Time of Day)

The environment simulates a full day cycle:
- **Steps 0-39:** Night — arrival rate halved, quiet roads
- **Steps 40-119:** Normal — standard traffic
- **Steps 120-159:** Rush Hour — arrival rate doubled, rewards doubled
- **Steps 160-199:** Back to Normal

This matters because agents must learn different strategies for different times of day.

## Emergency Vehicles

Every 50 steps (Medium), 20 steps (Hard), or 10 steps (Expert), a random ambulance appears at a random intersection. If the agent does NOT give green to the ambulance's lane within 2 steps → penalty of -50 per step. This forces agents to learn emergency prioritization.

---

# 4. THE STATE VECTOR — WHAT EACH AGENT SEES

Each agent has **partial observability** — it cannot see the whole city, only:
- Its own 4 lane counts
- Its 2 neighbors' total car counts
- The current time of day
- Whether an ambulance is present at its intersection

This is represented as **8 numbers:**

```
state = [c_N, c_S, c_E, c_W, n_left, n_right, time_slot, emergency_flag]
```

| Position | Name | What it means | Range |
|---|---|---|---|
| 0 | c_N | Cars in North lane | 0-30 |
| 1 | c_S | Cars in South lane | 0-30 |
| 2 | c_E | Cars in East lane | 0-30 |
| 3 | c_W | Cars in West lane | 0-30 |
| 4 | n_left | Total cars at left neighbor | 0-120 |
| 5 | n_right | Total cars at right neighbor | 0-120 |
| 6 | time_slot | Time of day | 0=night, 1=normal, 2=rush |
| 7 | emergency_flag | Ambulance present | 0=no, 1=yes |

## Example State Explained

```
state = [10, 3, 7, 2, 18, 5, 2, 0]
```

Reading this: "My North lane has 10 cars, South has 3, East has 7, West has 2. My left neighbor has 18 total cars (congested), right neighbor has 5 (clear). It is rush hour. No ambulance."

**What should the agent do?** Give green to North (10 cars waiting) to clear the most congestion.

## Why Partial Observability?

Real traffic agents (sensors, cameras) cannot see the entire city. Each intersection only has local sensors. Our design is realistic — agents must make decisions based on limited information. This is harder but more valuable.

---

# 5. ACTIONS — WHAT EACH AGENT CAN DO

Each agent has exactly **4 possible actions** per step:

| Action | Meaning | When to use |
|---|---|---|
| 0 | Give green to North lane | North has most cars OR ambulance in North |
| 1 | Give green to South lane | South has most cars |
| 2 | Give green to East lane | East has most cars |
| 3 | Give green to West lane | West has most cars |

**One action per agent per step.** Only one lane can have green at a time (like real traffic lights — you cannot give green to all directions simultaneously).

**Emergency override:** If emergency_flag = 1, agent always chooses action 0 (North green = ambulance priority). This is hardcoded as a safety rule.

---

# 6. THE REWARD FORMULA — WHY IT IS DESIGNED THIS WAY

This is the most important technical design decision in our project. Every number in this formula has a reason.

## The Formula

```
reward = -own_cars - 0.3 × neighbor_cars - 50 × emergency × 2.0_if_rush_hour
```

## Breaking It Down — Each Part and Why

### Part 1: -sum(own_cars)
```
reward = -sum(own_cars)
```

**What it does:** Penalizes the agent for every car sitting at its intersection.
If own_cars = [10, 3, 7, 2] → sum = 22 → reward contribution = -22

**Why negative?** We want LESS cars = BETTER. So more cars = more negative reward = agent learns to avoid this.

**Why sum of all 4 lanes?** Because an agent should clear ALL lanes, not just one. If it only clears one lane and ignores others, the total still stays high.

### Part 2: -0.3 × sum(neighbor_cars)
```
reward -= 0.3 × sum(neighbor_cars)
```

**What it does:** Penalizes the agent for neighbor congestion too — but with less weight (0.3 = 30%).

**Why 0.3 specifically?**
- If it was 0.0: agents are purely selfish — they ignore neighbors completely
- If it was 1.0: agents care equally about neighbors as themselves — this causes confusion because they stop managing their own intersection properly
- 0.3 is the balance: "I mainly care about my own intersection, but I am aware of my neighbors and will help if I can"

**What this creates:** Cooperative behavior. Agent 0 learns that if it holds cars from flowing to Agent 1 (who is already congested), both intersections do better. This is emergent cooperation — not programmed, but learned.

### Part 3: -50 × emergency_red_flag
```
reward -= 50 if ambulance is at red light
```

**What it does:** Massive penalty if an ambulance is stuck waiting at a red light.

**Why 50?** This number is much larger than typical step rewards (-20 to -100). It forces the agent to treat emergencies as the highest priority. An agent that ignores an ambulance loses 50 points instantly — it quickly learns to always prioritize.

**Why emergency_red_flag specifically?** If the agent gives green to the ambulance's lane (action 0), the flag immediately becomes 0 and penalty stops. Agent is rewarded for quick response.

### Part 4: ×2.0 if rush_hour
```
reward *= 2.0 during rush hour (steps 120-159)
```

**What it does:** Doubles all penalties during rush hour.

**Why?** Rush hour is when traffic is most damaging. Agents need to learn that rush hour mistakes are twice as costly. This teaches them to be MORE careful and proactive during rush hours — clearing lanes before they get jammed rather than reacting after.

**Effect on learning:** Agents that learn good rush-hour strategies score significantly better than agents that use the same strategy all day.

## Total City Reward

```
total_city_reward = reward_0 + reward_1 + reward_2 + reward_3
```

All 4 agents' rewards are summed. This means if one agent causes problems for the whole city, the total score drops. The training objective is to maximize the city-wide total — this aligns all agents toward a common goal.

---

# 7. FEDERATED Q-LEARNING — OUR CORE INNOVATION

## The Problem It Solves

Without federation, each agent learns independently. Agent 0 might discover after 50 episodes that "during rush hour, serve East first." But Agent 3, on the other side of the city, has to discover this same thing from scratch over another 50 episodes. Duplicate work. Slow learning.

## How Federation Works

Every 10 episodes, each agent shares its Q-table with its direct neighbors and averages the values.

**Step by step:**

```
Episode 1-10:   Each agent learns independently from experience
Episode 10:     FEDERATION EVENT
                Agent 0 shares Q-table with Agent 1 and Agent 2
                Agent 1 shares Q-table with Agent 0 and Agent 3
                Agent 2 shares Q-table with Agent 0 and Agent 3
                Agent 3 shares Q-table with Agent 1 and Agent 2

                Each agent averages received Q-tables with their own:
                Agent 0's new Q-table = average(Q0, Q1, Q2)

Episode 11-20:  Each agent continues learning with enriched knowledge
Episode 20:     FEDERATION EVENT again
                ... and so on every 10 episodes
```

## Why Averaging Q-Tables Works

Each Q-table is a record of "in situation X, action Y gave reward Z." When Agent 0 has seen 100 situations and Agent 1 has seen 80 different situations, averaging gives Agent 0 knowledge of 180 situations — without experiencing them directly.

Good strategies discovered at one corner of the city propagate to all corners within 10 episodes.

## Why "Federated" Learning?

The term comes from Federated Learning in distributed machine learning — where multiple devices (phones, hospitals, cars) train locally and share model updates without sharing raw data. Our approach adapts this concept to Q-tables in a multi-agent RL setting.

**Real-world analogy:** Traffic departments in different districts of Mumbai sharing best practices every month. What works in Andheri gets adopted in Bandra.

## What the Reward Curve Shows

On our reward curve graph, every 10 episodes you see an orange vertical line (federation event). After each federation event, the reward often improves slightly — agents get smarter from shared knowledge. The overall trend is upward = city-wide improvement over training.

---

# 8. WHAT IS Q-LEARNING? (FROM SCRATCH)

## The Simplest Explanation

Imagine you are playing a game for the first time. You do not know the rules. You just try random moves and see what happens. When a move gives you points, you remember "in that situation, that move was good." When it loses points, you remember "avoid that."

After thousands of tries, you have a mental map: "In situation A → do action 2. In situation B → do action 0." This mental map is the Q-table.

## The Q-Table

A Q-table is a dictionary:
```
Key:   (state_tuple)         → describes the current situation
Value: [Q0, Q1, Q2, Q3]      → expected reward for each action
```

Example:
```python
q_table[(2, 1, 3, 0, 1, 2, 1, 0)] = [-45.2, -38.1, -41.5, -50.3]
```

Reading this: "In this situation (moderate North, light South, heavy East, etc., normal time, no emergency), action 1 (South green) has the best expected reward of -38.1. Action 3 (West green) is worst at -50.3."

The agent picks action 1 because it has the highest (least negative) Q-value.

## How Q-Values Get Updated

After each step, the agent updates its Q-table using the Bellman equation:

```
Q(s, a) = Q(s, a) + lr × [reward + gamma × max(Q(s')) - Q(s, a)]
```

In plain English:
```
New Q-value = Old Q-value + learning_rate × (what actually happened - what we expected)
```

If reality was better than expected → Q-value increases → this action is preferred more
If reality was worse than expected → Q-value decreases → this action is preferred less

## Epsilon-Greedy Exploration

Early in training, the agent needs to TRY different actions to learn what works. If it always picks the best known action, it never discovers potentially better actions.

**Epsilon (ε) controls this:**
- At episode 1: ε = 1.0 → 100% random (pure exploration)
- At episode 100: ε = 0.45 → 45% random, 55% best known
- At episode 200: ε = 0.20 → 20% random, 80% best known

As training progresses, the agent gradually shifts from exploration to exploitation. This is called **epsilon decay**.

## State Encoding

The raw observation has numbers like [10, 3, 7, 2, 18, 5, 2, 0]. We cannot use these directly as Q-table keys because there are too many combinations (30^4 × 120^2 × 3 × 2 = billions).

We "discretize" (bin) the values:
```
0-5 cars   → bin 0 (empty)
6-10 cars  → bin 1 (light)
11-18 cars → bin 2 (moderate)
19-25 cars → bin 3 (heavy)
26+ cars   → bin 4 (jammed)
```

So [10, 3, 7, 2, 18, 5, 2, 0] becomes (1, 0, 1, 0, 2, 0, 2, 0) → a manageable Q-table key.

This gives us 5^6 × 3 × 2 = 56,250 possible states — a manageable size that still captures all important situations.

---

# 9. FILE-BY-FILE EXPLANATION

## ROOT FOLDER FILES

### models.py — Data Definitions
**What it does:** Defines the 3 data types the system uses.
**Think of it as:** The blueprint for all information that flows in the system.

**3 classes:**

**TrafficAction** — what an agent sends to the environment
```python
action = TrafficAction(agent_id=0, phase=2)
# Agent 0 wants to give green to East lane
```

**TrafficObservation** — what the environment sends back to one agent
```python
obs = TrafficObservation(
    agent_id=0,
    lane_counts=[10, 3, 7, 2],      # own lanes
    neighbor_totals=[18, 5],         # neighbor totals
    time_slot=2,                     # rush hour
    emergency_flag=0,                # no ambulance
    reward=-22.0,                    # got -22 this step
    done=False                       # episode not over
)
```

**CityState** — full city snapshot (all 4 intersections at once)
```python
state = CityState(
    step=45,
    all_lane_counts=[[10,3,7,2],[5,1,8,0],[3,9,2,4],[6,0,3,5]],
    current_phases=[2, 0, 1, 3],
    time_slot=2,
    emergency_flags=[0, 0, 1, 0],    # ambulance at intersection 2
    episode_reward=-450.0
)
```

**Why Pydantic?** Pydantic automatically validates data types. If someone sends agent_id="hello" instead of an integer, Pydantic immediately gives a clear error. This prevents mysterious bugs.

**Why inherits from OpenEnv base classes?** OpenEnv framework requires this. Without it, the framework cannot recognize our types and the server will not work. This is a mandatory hackathon requirement.

---

### agent.py — Learning Logic
**What it does:** Contains all agent learning code.
**Think of it as:** The brain of each traffic agent.

**2 classes:**

**QLearningAgent** — one agent for one intersection
- Has a Q-table (dictionary of state → Q-values)
- `encode_state()` — converts 8-number observation to Q-table key
- `select_action()` — epsilon-greedy action selection
- `learn()` — updates Q-table after each step
- `decay_epsilon()` — reduces exploration rate
- `save() / load()` — saves Q-table to JSON file

**FederatedAgents** — manages all 4 agents together
- Creates 4 QLearningAgents
- `select_actions()` — gets one action from each agent
- `learn_step()` — updates all 4 agents from one step
- `end_episode()` — decays epsilon, triggers federation
- `_federate()` — THE CORE INNOVATION: averages Q-tables between neighbors
- `save_all() / load_all()` — saves/loads all 4 agents

**The federation code in detail:**
```python
def _federate(self):
    # Take snapshots BEFORE changing anything
    snapshots = {i: dict(agent.q_table) for i, agent in enumerate(self.agents)}

    for i, agent in enumerate(self.agents):
        neighbor_ids = self.NEIGHBORS[i]  # which agents are neighbors

        # Find all states seen by this agent OR any neighbor
        all_states = set(snapshots[i].keys())
        for n in neighbor_ids:
            all_states.update(snapshots[n].keys())

        # Average Q-values for each state
        new_table = {}
        for state in all_states:
            tables_for_state = [snapshots[i][state]] if state in snapshots[i] else []
            for n in neighbor_ids:
                if state in snapshots[n]:
                    tables_for_state.append(snapshots[n][state])
            new_table[state] = np.mean(tables_for_state, axis=0)

        agent.q_table = new_table
```

**Why snapshot first?** If we modified Agent 0's table and then used it to update Agent 1, the averaging would be unfair — Agent 1 would get Agent 0's already-modified values, not original values. Taking snapshots first ensures fair averaging.

---

### server/smartcity_traffic_environment.py — The City Simulation
**What it does:** The heart of the project. Simulates the entire city.
**Think of it as:** The actual city that agents interact with.

**Inherits from:** `openenv.core.env_server.Environment` (OpenEnv mandatory)

**Key methods:**

`reset(seed, episode_id)` → Resets city to initial state. Returns first observation for Agent 0. Called at start of each training episode.

`step(action: TrafficAction)` → Takes ONE agent's action. Buffers it. When all 4 agents have submitted → advances the full simulation. Returns that agent's observation.

`state` (property) → Returns CityState — full city snapshot. OpenEnv requires this to be a property, not a method.

**The simulation pipeline (inside `_advance_simulation`):**
```
1. Apply all 4 agents' actions (update green light phases)
2. Update time slot (night/normal/rush)
3. Trigger emergencies if scheduled
4. Discharge green lanes (cars leave)
5. Flow cars between intersections (THE MULTI-AGENT PART)
6. Add new arriving cars from outside city
7. Clamp all lanes to max 30
8. Calculate rewards for all agents
9. Build observations for all agents
```

**Why buffer actions?** OpenEnv calls `step()` once per agent. But the city should only advance once ALL agents have decided. So we collect all 4 actions first, then advance. This ensures synchronized multi-agent stepping.

---

### server/app.py — The Web Server
**What it does:** Wraps the environment as a web API.
**Think of it as:** The reception desk — agents send requests here to interact with the environment.

**Key line:**
```python
app = create_fastapi_app(
    env             = lambda: CityTrafficEnvironment(task="easy"),
    action_cls      = TrafficAction,
    observation_cls = TrafficObservation,
)
```

`create_fastapi_app` is OpenEnv's factory function. It automatically creates all standard endpoints:
- `POST /reset` — start new episode
- `POST /step` — send one action
- `GET /state` — see full city
- `GET /health` — server alive check
- `GET /docs` — interactive API explorer

**Why lambda?** `create_fastapi_app` needs a factory (callable) to create environment instances, not an instance directly. `lambda: CityTrafficEnvironment()` is a function that creates a new instance when called.

---

### train.py — Training Loop
**What it does:** Runs the complete training process.
**Think of it as:** The gym session where agents practice.

**Flow:**
```
Create environment + 4 agents
For each of 200 episodes:
    Reset environment
    Get all 4 agents' initial observations
    While not done:
        All 4 agents pick actions simultaneously
        Step all 4 agents through environment
        Collect rewards from each step
        All agents learn from what happened
    End of episode: decay epsilon, maybe federate
    Every 10 episodes: print progress
Training complete: save agents + plot reward curve
```

**Output files:**
- `saved_agents/agent_0.json` through `agent_3.json` — trained Q-tables
- `reward_curve.png` — proof of learning for judges
- `training_results.json` — summary numbers

---

### compare.py — Comparison Graph
**What it does:** Runs 3 different agents and plots them together.
**Think of it as:** The proof that our innovation works.

**3 agents compared:**
1. **Random agent** — picks random actions, baseline
2. **Q-Learning without federation** — learns but does not share
3. **Federated Q-Learning** — our full system

**Output:** `comparison_graph.png` — shows GREEN (federated) beats BLUE (no federation) beats RED (random). This is the single most convincing visual proof of our innovation.

---

### demo.py — Live Terminal Demo
**What it does:** Shows the city grid updating in real time.
**Think of it as:** The live performance during the pitch.

**What it shows:**
- Each intersection with lane counts as a bar chart
- Current green light direction
- Congestion level (✅ Clear / 🟡 Light / 🟠 Moderate / 🔴 Heavy / 💀 JAMMED)
- Emergency ambulance alerts (🚨)
- City-wide total cars and running reward

**How to run during pitch:**
```cmd
python demo.py --task medium --steps 20
```

Use `--task medium` for the pitch — it shows emergencies and rush hour which is more impressive than easy.

---

### inference.py — Run Trained Agent
**What it does:** Loads trained agents and runs one complete episode.
**Think of it as:** The final exam — testing trained agents for real.

**Why mandatory?** Hackathon rules require this file. Judges run it to verify agents actually learned something.

**Two modes:**
```cmd
python inference.py --standalone --task medium   # direct, no server
python inference.py --task medium                # through server
```

---

### client.py — HTTP Client
**What it does:** Lets external code talk to the running server.
**Think of it as:** The telephone — agents use this to call the server.

**Main methods:**
- `reset(task)` → start new episode
- `step(agent_id, phase)` → send action
- `get_state()` → see full city
- `health()` → check server is alive

---

### openenv.yaml — Framework Configuration
**What it does:** Tells OpenEnv framework about our environment.
**Think of it as:** The registration form for our environment.

**Key fields:**
- `name` — environment identifier
- `class` — where to find our CityTrafficEnvironment class
- `action_type` / `observation_type` — our model classes
- `base_image` — Docker base image (must be openenv-base)

**Why required?** OpenEnv uses this file to understand your environment structure when deploying to HuggingFace Spaces.

---

### Dockerfile — Container Definition
**What it does:** Defines how to package everything into a Docker container.
**Think of it as:** A recipe for building a portable box that contains our entire project.

**Key line:**
```dockerfile
FROM ghcr.io/meta-pytorch/openenv-base:latest
```

**Why openenv-base?** This is a hackathon requirement. The base image includes all OpenEnv dependencies pre-installed, ensuring our environment runs identically everywhere.

**What Docker does:** Takes all our Python files, installs all dependencies, and creates a self-contained unit that runs anywhere — your laptop, Meta's servers, HuggingFace Spaces.

---

### SmartCity_TRL_Training.ipynb — LLM Training Notebook
**What it does:** Trains a language model on our environment using HF TRL.
**Think of it as:** The bridge between our Q-Learning environment and the LLM world.

**The innovation:** Instead of a Q-table deciding actions, a language model reads the traffic state as text and outputs a decision.

**How it works:**
```
Traffic state → Text prompt:
"You are agent 0. North=10, South=3, East=7, West=2.
 Neighbor left=18, right=5. Rush hour. No ambulance.
 Choose 0=North 1=South 2=East 3=West. Answer:"

LLM reads prompt → generates response → "2"

Parse "2" → action = 2 (East green)

Run in environment → get reward

TRL updates LLM weights based on reward
```

**Why GRPO?** Group Relative Policy Optimization is the algorithm Meta uses to train LLaMA. We use the same algorithm. This is directly relevant to Meta engineers and shows we understand modern LLM training.

**Result:** LLM achieved 100% accuracy on 4 test scenarios — correctly giving green to the busiest lane and always prioritizing ambulances.

---

# 10. HOW OPENENV FRAMEWORK WORKS

## What OpenEnv Is

OpenEnv is Meta's open-source framework for building standardized RL environments. Think of it as a standard plug-socket system — any environment built with OpenEnv can be used by any RL training framework (TRL, SkyRL, Oumi etc.) without any custom integration.

## The 3 Required Methods

Every OpenEnv environment must implement exactly 3 things:

```python
def reset(self, seed=None, episode_id=None) -> Observation:
    # Reset to start state, return first observation
    pass

def step(self, action: Action) -> Observation:
    # Apply action, return new observation
    pass

@property
def state(self) -> State:
    # Return full current state
    pass
```

Our `CityTrafficEnvironment` implements all 3 correctly.

## The Deployment Pipeline

```
Local code
    ↓
Docker container (openenv-base image)
    ↓
HuggingFace Spaces
    ↓
Live URL: https://username-smartcity-traffic.hf.space
    ↓
Any agent can connect via HTTP and train on our environment
```

---

# 11. HF TRL — HOW WE TRAIN AN LLM

## What TRL Is

TRL (Transformer Reinforcement Learning) is HuggingFace's library for training language models using reinforcement learning. It implements algorithms like GRPO (what Meta used for LLaMA) and PPO.

## How We Connect TRL to Our Environment

```python
def compute_reward(prompts, completions):
    # TRL calls this after LLM generates a response
    # We run the response in our environment
    # Return the environment's reward as the training signal
    action  = parse_action(completions[0], current_obs)
    obs     = env.step(TrafficAction(agent_id=0, phase=action))
    return [obs.reward / 200.0]   # normalized reward
```

This `compute_reward` function is the bridge. TRL trains the LLM to generate responses that maximize this reward. Our environment IS the reward signal.

---

# 12. KEY TECHNICAL TERMS EXPLAINED SIMPLY

| Term | Simple Explanation |
|---|---|
| **Reinforcement Learning** | Learning by trying things and getting rewards/penalties |
| **Q-Learning** | A specific RL algorithm using a lookup table |
| **Q-table** | A dictionary mapping situations to expected rewards |
| **Episode** | One complete run of 200 steps (one simulated day) |
| **Step** | One decision point — all 4 agents act once |
| **Reward** | Score for each step (negative = bad, less negative = better) |
| **State** | Description of current situation (8 numbers per agent) |
| **Action** | What an agent does (0=North green, 1=South, 2=East, 3=West) |
| **Epsilon** | Probability of random exploration (starts 1.0, decays to 0.05) |
| **Epsilon decay** | Gradually reducing randomness as agent gets smarter |
| **Discount (gamma)** | How much future rewards matter (0.95 = care a lot about future) |
| **Learning rate** | How fast Q-values update (0.1 = moderate speed) |
| **Federation** | Sharing Q-tables between neighbors every 10 episodes |
| **OpenEnv** | Meta's framework for building standard RL environments |
| **TRL** | HuggingFace's library for training LLMs with RL |
| **GRPO** | Algorithm used to train LLaMA — we use same for our LLM |
| **Pydantic** | Python library that validates data types automatically |
| **FastAPI** | Web framework that creates HTTP endpoints automatically |
| **Docker** | Packages entire project into portable container |
| **HF Spaces** | HuggingFace's cloud hosting for ML demos |
| **Partial observability** | Agent cannot see full world — only local information |

---

---

# APPENDIX: QUICK REFERENCE NUMBERS

| Metric | Value |
|---|---|
| Intersections | 4 (2×2 grid) |
| Lanes total | 16 |
| Max cars per lane | 30 |
| State vector size | 8 numbers per agent |
| Actions available | 4 (N/S/E/W green) |
| Episode length | 200 steps |
| Training episodes | 200 |
| Federation interval | Every 10 episodes |
| Federation events | 20 total |
| Reward improvement | +7,000 to +10,000 (varies per run) |
| LLM accuracy |75% on test scenarios (Qwen2.5-0.5B on GPU) |
| Emergency penalty | -50 per step |
| Rush hour multiplier | 2.0× |
| Cooperative weight | 0.3 (30%) |
| Epsilon start | 1.0 (100% random) |
| Epsilon end | ~0.2 (20% random) |
| Learning rate | 0.1 |
| Discount (gamma) | 0.95 |
| Q-table states learned | ~1,183 per agent |
| Training time (CPU) | ~8 seconds |

---
