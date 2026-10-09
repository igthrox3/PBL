import os
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Circle, Rectangle

from traffic_env import TrafficEnv, PHASE_NS


st.set_page_config(page_title="Smart Traffic Signal Control", layout="wide")


SCENARIOS = {
    "Low":    (0.05, 0.05, 0.10, 0.10),
    "Normal": (0.10, 0.10, 0.25, 0.25),
    "High":   (0.20, 0.20, 0.40, 0.40),
}

AGENT_LABEL = {
    "dqn":         "DQN (Deep RL)",
    #"qlearning":   "Tabular Q-learning",
    "fixed_timer": "Fixed timer (baseline)",
}


@st.cache_resource
def load_agent(name, ckpt_dir="checkpoints"):
    env = TrafficEnv()
    if name == "fixed_timer":
        from agents.fixed_timer import FixedTimerAgent
        return FixedTimerAgent(period=20)
    '''if name == "qlearning":
        from agents.q_learning import QLearningAgent
        a = QLearningAgent(env.n_actions)
        a.load(os.path.join(ckpt_dir, "qlearning.pkl"))
        a.eps = 0.0
        return a'''
    if name == "dqn":
        from agents.dqn import DQNAgent
        a = DQNAgent(env.state_dim, env.n_actions)
        a.load(os.path.join(ckpt_dir, "dqn.pt"))
        a.eps = 0.0
        return a
    raise ValueError(name)


def render_intersection(env, info):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("#2a5a32")
    ax.set_facecolor("#2a5a32")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")

    cx, cy = 50, 50
    road_w = 18

    ax.add_patch(Rectangle((cx - road_w / 2, 0), road_w, 100, color="#3c3c3c"))
    ax.add_patch(Rectangle((0, cy - road_w / 2), 100, road_w, color="#3c3c3c"))

    for y in np.arange(0, 100, 4):
        ax.add_patch(Rectangle((cx - 0.25, y), 0.5, 2, color="white"))
    for x in np.arange(0, 100, 4):
        ax.add_patch(Rectangle((x, cy - 0.25), 2, 0.5, color="white"))

    ax.add_patch(Rectangle((cx - road_w / 2, cy - road_w / 2), road_w, road_w, color="#1e1e1e"))

    is_yellow = info["yellow"]
    ns_green = (env.phase == PHASE_NS) and not is_yellow
    ew_green = (env.phase != PHASE_NS) and not is_yellow

    def light_color(is_green):
        if is_yellow:
            return "#f0c020"
        return "#28d860" if is_green else "#d83838"

    ax.add_patch(Circle((cx - road_w, cy + road_w), 1.6, color=light_color(ns_green)))
    ax.add_patch(Circle((cx + road_w, cy - road_w), 1.6, color=light_color(ns_green)))
    ax.add_patch(Circle((cx + road_w, cy + road_w), 1.6, color=light_color(ew_green)))
    ax.add_patch(Circle((cx - road_w, cy - road_w), 1.6, color=light_color(ew_green)))

    car_color = "#f0c83c"
    cw, ch = 2.4, 3.6
    q = info["queues"]
    max_show = 22

    for i in range(min(int(q[0]), max_show)):
        y = cy + road_w / 2 + 1 + i * (ch + 0.4)
        if y > 100:
            break
        ax.add_patch(Rectangle((cx - 4 - cw / 2, y), cw, ch, color=car_color))

    for i in range(min(int(q[1]), max_show)):
        y = cy - road_w / 2 - 1 - (i + 1) * (ch + 0.4)
        if y < 0:
            break
        ax.add_patch(Rectangle((cx + 4 - cw / 2, y), cw, ch, color=car_color))

    for i in range(min(int(q[2]), max_show)):
        x = cx + road_w / 2 + 1 + i * (ch + 0.4)
        if x > 100:
            break
        ax.add_patch(Rectangle((x, cy + 4 - cw / 2), ch, cw, color=car_color))

    for i in range(min(int(q[3]), max_show)):
        x = cx - road_w / 2 - 1 - (i + 1) * (ch + 0.4)
        if x < 0:
            break
        ax.add_patch(Rectangle((x, cy - 4 - cw / 2), ch, cw, color=car_color))

    fig.tight_layout(pad=0.5)
    return fig


def main():
    st.title("Smart Traffic Signal Control")
    st.caption("A reinforcement learning agent controls signal timing at a 4-way intersection.")

    st.sidebar.header("Configuration")
    agent_options = []
    if os.path.exists("checkpoints/dqn.pt"):
        agent_options.append("dqn")
    """if os.path.exists("checkpoints/qlearning.pkl"):
        agent_options.append("qlearning")"""
    agent_options.append("fixed_timer")

    agent_name = st.sidebar.selectbox(
        "Agent",
        agent_options,
        format_func=lambda s: AGENT_LABEL[s],
    )
    scenario = st.sidebar.selectbox("Traffic scenario", list(SCENARIOS.keys()), index=1)
    steps = st.sidebar.slider("Simulation steps", 100, 1000, 500, 100)
    speed = st.sidebar.slider("Speed (steps/sec)", 1, 30, 10)
    seed = st.sidebar.number_input("Random seed", value=42, step=1)

    start = st.sidebar.button("Start simulation", type="primary", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**About**\n\n"
        "DQN agent trained with PyTorch reduces avg queue length by ~52% "
        "and waiting time by ~61% versus a fixed-timer baseline."
    )

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Intersection")
        viz_box = st.empty()

    with right:
        st.subheader("Live metrics")
        metric_box = st.empty()
        st.subheader("Total queue over time")
        chart_box = st.empty()
        st.subheader("Action log")
        log_box = st.empty()

    if not start:
        with left:
            viz_box.info("Pick an agent and traffic scenario in the sidebar, then click **Start simulation**.")
        return

    env = TrafficEnv(max_steps=steps, arrival_rates=SCENARIOS[scenario])
    agent = load_agent(agent_name)
    state = env.reset(seed=int(seed))
    info = {"queues": env.queues, "yellow": False, "cars_passed": 0}

    queue_history = []
    action_log = []
    switch_count = 0
    final_m = env.metrics()

    for step in range(steps):
        action = agent.act(state, greedy=True)
        if action == 1:
            switch_count += 1
            action_log.append(f"step {step:4d}  SWITCH  queues={list(map(int, env.queues))}")

        state, _, done, info = env.step(action)
        queue_history.append(int(np.sum(env.queues)))
        final_m = env.metrics()

        fig = render_intersection(env, info)
        viz_box.pyplot(fig, clear_figure=True)
        plt.close(fig)

        phase_label = "EW" if env.phase != PHASE_NS else "NS"
        if info["yellow"]:
            phase_label += " (yellow)"
        action_label = "SWITCH" if action == 1 else "KEEP"

        with metric_box.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Step", f"{step + 1}/{steps}")
            c2.metric("Phase", phase_label)
            c3.metric("Action", action_label)
            c4.metric("Switches", switch_count)
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("N queue", int(env.queues[0]))
            c6.metric("S queue", int(env.queues[1]))
            c7.metric("E queue", int(env.queues[2]))
            c8.metric("W queue", int(env.queues[3]))
            c9, c10, c11, c12 = st.columns(4)
            c9.metric("Throughput", final_m["total_throughput"])
            c10.metric("Avg queue", f"{final_m['avg_queue']:.2f}")
            c11.metric("Avg wait", f"{final_m['avg_wait_per_step']:.2f}")
            c12.metric("Max queue", final_m["max_queue"])

        chart_box.line_chart(queue_history, height=200)

        if action_log:
            log_box.code("\n".join(action_log[-12:]))
        else:
            log_box.code("(agent has not switched yet)")

        time.sleep(1.0 / speed)

        if done:
            break

    st.success(
        f"Simulation complete. "
        f"Avg queue: {final_m['avg_queue']:.2f} | "
        f"Avg wait/step: {final_m['avg_wait_per_step']:.2f} | "
        f"Max queue: {final_m['max_queue']} | "
        f"Throughput: {final_m['total_throughput']} | "
        f"Switches: {switch_count}"
    )


if __name__ == "__main__":
    main()
