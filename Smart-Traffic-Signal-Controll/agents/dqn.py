import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(np.array(ns), dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, state_dim, n_actions, lr=1e-3, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.995,
                 batch_size=64, target_update=200, buffer_size=50_000,
                 device=None):
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = QNetwork(state_dim, n_actions).to(self.device)
        self.target = QNetwork(state_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(buffer_size)
        self.train_steps = 0

    def act(self, state, greedy=False):
        if not greedy and random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.policy(s).argmax(dim=1).item())

    def update(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, float(done))
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        s, a, r, ns, d = s.to(self.device), a.to(self.device), r.to(self.device), ns.to(self.device), d.to(self.device)

        q = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target(ns).max(dim=1).values
            target = r + self.gamma * q_next * (1.0 - d)

        loss = self.loss_fn(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.opt.step()

        self.train_steps += 1
        if self.train_steps % self.target_update == 0:
            self.target.load_state_dict(self.policy.state_dict())

        return float(loss.item())

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def save(self, path):
        torch.save({"policy": self.policy.state_dict(), "eps": self.eps}, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(data["policy"])
        self.target.load_state_dict(data["policy"])
        self.eps = data.get("eps", self.eps_end)
