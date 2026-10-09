import numpy as np
import pickle


class QLearningAgent:
    def __init__(self, n_actions=2, alpha=0.1, gamma=0.95, eps_start=1.0,
                 eps_end=0.05, eps_decay=0.995, queue_bins=4, max_queue=40):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.queue_bins = queue_bins
        self.max_queue = max_queue
        self.q_table = {}

    def _discretize(self, state):
        norm_q = state[:4]
        phase = int(state[4])
        bins = tuple(int(min(self.queue_bins - 1, q * self.queue_bins)) for q in norm_q)
        return bins + (phase,)

    def _q(self, key):
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions, dtype=np.float32)
        return self.q_table[key]

    def act(self, state, greedy=False):
        key = self._discretize(state)
        if not greedy and np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self._q(key)))

    def update(self, state, action, reward, next_state, done):
        key = self._discretize(state)
        next_key = self._discretize(next_state)
        target = reward
        if not done:
            target += self.gamma * np.max(self._q(next_key))
        q = self._q(key)
        q[action] += self.alpha * (target - q[action])

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"q_table": self.q_table, "eps": self.eps}, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.eps = data["eps"]
