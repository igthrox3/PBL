# Smart Traffic Signal Control with Reinforcement Learning

A reinforcement learning agent that controls traffic light timing at a 4-way intersection to minimize average queue length and waiting time.

![Dashboard](assets/dashboard.png)

## Results

DQN was evaluated against a fixed-timer baseline and a tabular Q-learning agent across three traffic scenarios (low, normal, high). All numbers are averaged over 20 episodes of 500 steps each.

| Scenario | Agent | Avg queue | Avg wait | Throughput |
|----------|-------|-----------|----------|------------|
| Low      | fixed_timer | 2.21 | 13.31 | 142.1 |
| Low      | dqn         | 1.03 |  4.10 | 144.5 |
| Normal   | fixed_timer | 6.10 | 27.02 | 334.8 |
| Normal   | dqn         | 2.95 | 10.41 | 341.2 |
| High     | fixed_timer | 18.31 | 164.98 | 569.5 |
| High     | dqn         |  8.92 |  63.02 | 586.5 |

DQN reduces average queue length by roughly 51-53% across all three scenarios.

## Project structure

```
traffic_env.py       4-way intersection simulator (Gym-style API)
agents/
  fixed_timer.py     baseline that switches phase every N seconds
  q_learning.py      tabular Q-learning agent
  dqn.py             DQN agent with replay buffer and target network
train.py             training loop for DQN or Q-learning
evaluate.py          multi-scenario comparison of all agents
plot_training.py     training curve plots
visualize.py         pygame animation of the intersection
app.py               Streamlit web dashboard (live intersection + metrics)
```

## RL formulation

State (6 dims): normalized queue lengths for N, S, E, W lanes, current phase, and time spent in current phase.

Actions (2): keep current phase, or switch (subject to a 5-second minimum green and a 3-second yellow transition).

Reward: `-(total_queue) - 0.1 * total_waiting_time`

Dynamics: Poisson arrivals per lane (asymmetric by default, with E/W busier than N/S), 1 car/sec departure on green lanes.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train DQN (about 1-2 minutes on CPU)
python train.py --agent dqn --episodes 300

# Train tabular Q-learning baseline
python train.py --agent qlearning --episodes 300

# Compare all agents across low/normal/high traffic
python evaluate.py --episodes 20

# Plot training curves
python plot_training.py --agent dqn

# Watch the trained agent (pygame desktop window)
python visualize.py --agent dqn

# Run the Streamlit web dashboard (opens at http://localhost:8501)
streamlit run app.py
```

## Web dashboard

The Streamlit app provides an interactive browser interface to:

- Pick the agent (DQN, Q-learning, or Fixed timer)
- Pick a traffic scenario (Low, Normal, High)
- Watch the intersection live with cars queueing on each lane
- See real-time metrics (queue per lane, throughput, avg wait, max queue, switch count)
- Track total queue length over time on a live chart
- View the most recent SWITCH actions in an action log

## Notes

The tabular Q-learning agent performs poorly in this setup because discretizing four continuous queue counts into bins loses too much information. DQN handles the continuous state directly via an MLP and generalizes across similar traffic patterns.

The minimum green time and yellow transition are required to prevent the agent from oscillating the phase rapidly, which would not be physically realistic.
