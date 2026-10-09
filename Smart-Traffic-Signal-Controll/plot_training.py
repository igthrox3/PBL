import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def plot(agent_name="dqn", ckpt_dir="checkpoints"):
    log_path = os.path.join(ckpt_dir, f"{agent_name}_log.npz")
    data = np.load(log_path)
    rewards, queues, waits = data["rewards"], data["avg_queue"], data["avg_wait"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(rewards)
    axes[0].set_title("Episode reward")
    axes[0].set_xlabel("episode")
    axes[1].plot(queues, color="tab:orange")
    axes[1].set_title("Average queue length")
    axes[1].set_xlabel("episode")
    axes[2].plot(waits, color="tab:green")
    axes[2].set_title("Average waiting time / step")
    axes[2].set_xlabel("episode")
    fig.suptitle(f"Training curves — {agent_name}")
    fig.tight_layout()

    out = os.path.join(ckpt_dir, f"{agent_name}_curves.png")
    fig.savefig(out, dpi=120)
    print(f"saved -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["dqn", "qlearning"], default="dqn")
    args = p.parse_args()
    plot(args.agent)
