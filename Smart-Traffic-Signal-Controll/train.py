import argparse
import os

import numpy as np

from traffic_env import TrafficEnv


def train(agent_name="dqn", episodes=300, steps_per_episode=500,
          save_dir="checkpoints", log_every=1):
    os.makedirs(save_dir, exist_ok=True)
    env = TrafficEnv(max_steps=steps_per_episode, seed=0)

    if agent_name == "dqn":
        from agents.dqn import DQNAgent
        agent = DQNAgent(env.state_dim, env.n_actions)
        save_path = os.path.join(save_dir, "dqn.pt")
    elif agent_name == "qlearning":
        from agents.q_learning import QLearningAgent
        agent = QLearningAgent(env.n_actions)
        save_path = os.path.join(save_dir, "qlearning.pkl")
    else:
        raise ValueError(f"unknown agent: {agent_name}")

    rewards_log, queue_log, wait_log = [], [], []

    for ep in range(episodes):
        state = env.reset(seed=ep)
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

        agent.decay_epsilon()
        m = env.metrics()
        rewards_log.append(ep_reward)
        queue_log.append(m["avg_queue"])
        wait_log.append(m["avg_wait_per_step"])

        if (ep + 1) % log_every == 0:
            recent = np.mean(rewards_log[-log_every:])
            recent_q = np.mean(queue_log[-log_every:])
            print(
                f"ep {ep+1:4d} | reward {recent:9.1f} | avg_queue {recent_q:5.2f} "
                f"| eps {agent.eps:.3f} | throughput {m['total_throughput']}"
            )

    agent.save(save_path)
    print(f"saved -> {save_path}")

    log_path = os.path.join(save_dir, f"{agent_name}_log.npz")
    np.savez(log_path, rewards=rewards_log, avg_queue=queue_log, avg_wait=wait_log)
    print(f"logs  -> {log_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["dqn", "qlearning"], default="dqn")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--steps", type=int, default=500)
    args = p.parse_args()
    train(args.agent, args.episodes, args.steps)
