import argparse
import os

import pygame

from traffic_env import TrafficEnv, PHASE_NS

WIDTH, HEIGHT = 700, 700
ROAD_COLOR = (60, 60, 60)
GRASS = (40, 90, 50)
LANE_LINE = (220, 220, 220)
CAR_COLOR = (240, 200, 60)
GREEN_LIGHT = (40, 220, 90)
RED_LIGHT = (220, 60, 60)
YELLOW_LIGHT = (240, 200, 60)
TEXT = (240, 240, 240)


def load_agent(name, env, ckpt_dir="checkpoints"):
    if name == "fixed_timer":
        from agents.fixed_timer import FixedTimerAgent
        return FixedTimerAgent(period=20)
    if name == "qlearning":
        from agents.q_learning import QLearningAgent
        a = QLearningAgent(env.n_actions)
        a.load(os.path.join(ckpt_dir, "qlearning.pkl"))
        a.eps = 0.0
        return a
    if name == "dqn":
        from agents.dqn import DQNAgent
        a = DQNAgent(env.state_dim, env.n_actions)
        a.load(os.path.join(ckpt_dir, "dqn.pt"))
        a.eps = 0.0
        return a
    raise ValueError(name)


def draw_intersection(screen, font, env, info, last_action, last_action_label):
    screen.fill(GRASS)
    cx, cy = WIDTH // 2, HEIGHT // 2
    road_w = 120

    pygame.draw.rect(screen, ROAD_COLOR, (cx - road_w // 2, 0, road_w, HEIGHT))
    pygame.draw.rect(screen, ROAD_COLOR, (0, cy - road_w // 2, WIDTH, road_w))

    for y in range(0, HEIGHT, 30):
        pygame.draw.rect(screen, LANE_LINE, (cx - 2, y, 4, 15))
    for x in range(0, WIDTH, 30):
        pygame.draw.rect(screen, LANE_LINE, (x, cy - 2, 15, 4))

    pygame.draw.rect(screen, (30, 30, 30), (cx - road_w // 2, cy - road_w // 2, road_w, road_w))

    is_yellow = info["yellow"]
    ns_green = (env.phase == PHASE_NS) and not is_yellow
    ew_green = (env.phase != PHASE_NS) and not is_yellow

    def light_color(is_green):
        if is_yellow:
            return YELLOW_LIGHT
        return GREEN_LIGHT if is_green else RED_LIGHT

    pygame.draw.circle(screen, light_color(ns_green), (cx - road_w, cy - road_w), 12)
    pygame.draw.circle(screen, light_color(ns_green), (cx + road_w, cy + road_w), 12)
    pygame.draw.circle(screen, light_color(ew_green), (cx + road_w, cy - road_w), 12)
    pygame.draw.circle(screen, light_color(ew_green), (cx - road_w, cy + road_w), 12)

    car_w, car_h = 16, 26
    queues = info["queues"]

    for i in range(queues[0]):
        y = cy - road_w // 2 - 30 - i * (car_h + 4)
        if y < 0:
            break
        pygame.draw.rect(screen, CAR_COLOR, (cx - 30 - car_w // 2, y, car_w, car_h))

    for i in range(queues[1]):
        y = cy + road_w // 2 + 30 + i * (car_h + 4)
        if y > HEIGHT:
            break
        pygame.draw.rect(screen, CAR_COLOR, (cx + 30 - car_w // 2, y, car_w, car_h))

    for i in range(queues[2]):
        x = cx + road_w // 2 + 30 + i * (car_h + 4)
        if x > WIDTH:
            break
        pygame.draw.rect(screen, CAR_COLOR, (x, cy + 30 - car_w // 2, car_h, car_w))

    for i in range(queues[3]):
        x = cx - road_w // 2 - 30 - i * (car_h + 4)
        if x < 0:
            break
        pygame.draw.rect(screen, CAR_COLOR, (x, cy - 30 - car_w // 2, car_h, car_w))

    action_color = (240, 200, 60) if last_action == 1 else (180, 220, 240)
    lines = [
        f"step {env.step_count} / {env.max_steps}",
        f"phase: {'NS' if env.phase == PHASE_NS else 'EW'} {'(yellow)' if is_yellow else ''}",
        f"queues  N:{queues[0]} S:{queues[1]} E:{queues[2]} W:{queues[3]}",
        f"max queue seen: {env.max_queue_seen}",
        f"throughput: {info['cars_passed']}",
    ]
    for i, line in enumerate(lines):
        surf = font.render(line, True, TEXT)
        screen.blit(surf, (10, 10 + i * 22))

    action_surf = font.render(f"action: {last_action_label}", True, action_color)
    screen.blit(action_surf, (10, 10 + len(lines) * 22 + 6))


def main(agent_name="dqn", steps=500, fps=10):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Traffic Signal — {agent_name}")
    font = pygame.font.SysFont("monospace", 16)
    clock = pygame.time.Clock()

    env = TrafficEnv(max_steps=steps, seed=42)
    agent = load_agent(agent_name, env)
    state = env.reset(seed=42)
    info = {"queues": env.queues, "yellow": False, "cars_passed": 0}
    last_action = 0
    last_action_label = "KEEP"
    switch_count = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        action = agent.act(state, greedy=True)
        if action == 1:
            label = "SWITCH"
            switch_count += 1
            print(f"[step {env.step_count:4d}] SWITCH  (queues={list(env.queues)})")
        else:
            label = "KEEP"
        last_action, last_action_label = action, label

        state, _, done, info = env.step(action)
        draw_intersection(screen, font, env, info, last_action, last_action_label)
        pygame.display.flip()
        clock.tick(fps)
        if done:
            running = False

    print(f"total switches: {switch_count}")

    pygame.quit()
    print("metrics:", env.metrics())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["fixed_timer", "qlearning", "dqn"], default="dqn")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args()
    main(args.agent, args.steps, args.fps)
