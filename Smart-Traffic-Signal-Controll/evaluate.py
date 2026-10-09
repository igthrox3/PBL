import argparse
import os

import numpy as np

from traffic_env import TrafficEnv


SCENARIOS = {
    "low":    (0.05, 0.05, 0.10, 0.10),
    "normal": (0.10, 0.10, 0.25, 0.25),
    "high":   (0.20, 0.20, 0.40, 0.40),
}


def run_episode(env, agent, seed):
    state = env.reset(seed=seed)
    total_reward = 0.0
    done = False
    while not done:
        action = agent.act(state, greedy=True)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward, env.metrics()


def load_agents(env, ckpt_dir):
    from agents.fixed_timer import FixedTimerAgent
    agents = {"fixed_timer": FixedTimerAgent(period=20)}

    q_path = os.path.join(ckpt_dir, "qlearning.pkl")
    if os.path.exists(q_path):
        from agents.q_learning import QLearningAgent
        ql = QLearningAgent(env.n_actions)
        ql.load(q_path)
        ql.eps = 0.0
        agents["qlearning"] = ql

    dqn_path = os.path.join(ckpt_dir, "dqn.pt")
    if os.path.exists(dqn_path):
        from agents.dqn import DQNAgent
        dqn = DQNAgent(env.state_dim, env.n_actions)
        dqn.load(dqn_path)
        dqn.eps = 0.0
        agents["dqn"] = dqn

    return agents


def evaluate(n_episodes=20, steps_per_episode=500, ckpt_dir="checkpoints"):
    base_env = TrafficEnv(max_steps=steps_per_episode)
    agents = load_agents(base_env, ckpt_dir)
    agent_names = list(agents.keys())

    all_results = {}

    for scenario_name, rates in SCENARIOS.items():
        env = TrafficEnv(max_steps=steps_per_episode, arrival_rates=rates)
        results = {name: {"reward": [], "avg_queue": [], "avg_wait": [],
                          "throughput": [], "max_queue": []}
                   for name in agent_names}

        for ep in range(n_episodes):
            seed = 10_000 + ep
            for name, agent in agents.items():
                r, m = run_episode(env, agent, seed)
                results[name]["reward"].append(r)
                results[name]["avg_queue"].append(m["avg_queue"])
                results[name]["avg_wait"].append(m["avg_wait_per_step"])
                results[name]["throughput"].append(m["total_throughput"])
                results[name]["max_queue"].append(m["max_queue"])

        all_results[scenario_name] = results
        print_table(scenario_name, results)

    plot_comparison(all_results, ckpt_dir)


def print_table(scenario, results):
    print(f"\n=== Scenario: {scenario.upper()} traffic ===")
    print(f"{'agent':<14} {'reward':>10} {'avg_queue':>10} {'max_queue':>10} {'avg_wait':>10} {'throughput':>12}")
    print("-" * 72)
    for name, data in results.items():
        print(
            f"{name:<14} {np.mean(data['reward']):>10.1f} "
            f"{np.mean(data['avg_queue']):>10.2f} "
            f"{np.mean(data['max_queue']):>10.1f} "
            f"{np.mean(data['avg_wait']):>10.2f} "
            f"{np.mean(data['throughput']):>12.1f}"
        )

    if "fixed_timer" in results:
        base_q = np.mean(results["fixed_timer"]["avg_queue"])
        for name, data in results.items():
            if name == "fixed_timer":
                continue
            q = np.mean(data["avg_queue"])
            improvement = (base_q - q) / base_q * 100
            print(f"  {name:<14} {improvement:+6.1f}% queue reduction vs fixed_timer")


def plot_comparison(all_results, ckpt_dir):
    import matplotlib.pyplot as plt

    metrics = [("avg_queue", "Avg queue length"),
               ("avg_wait", "Avg waiting time / step"),
               ("max_queue", "Max queue (single lane)"),
               ("throughput", "Total throughput")]

    scenarios = list(all_results.keys())
    agents = list(next(iter(all_results.values())).keys())
    colors = {"fixed_timer": "#888888", "qlearning": "#e07b39", "dqn": "#2ca02c"}

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    x = np.arange(len(scenarios))
    bar_width = 0.8 / len(agents)

    for ax, (key, label) in zip(axes, metrics):
        for i, name in enumerate(agents):
            values = [np.mean(all_results[s][name][key]) for s in scenarios]
            offset = (i - (len(agents) - 1) / 2) * bar_width
            bars = ax.bar(x + offset, values, bar_width, label=name,
                          color=colors.get(name, None))
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        ax.set_title(label)
        ax.set_xlabel("traffic scenario")
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(loc="upper left")
    fig.suptitle("Agent comparison across traffic scenarios", fontsize=14)
    fig.tight_layout()

    out = os.path.join(ckpt_dir, "comparison.png")
    fig.savefig(out, dpi=120)
    print(f"\ncomparison plot -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--steps", type=int, default=500)
    args = p.parse_args()
    evaluate(args.episodes, args.steps)
