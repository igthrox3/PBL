import numpy as np


PHASE_NS = 0
PHASE_EW = 1
PHASES = [PHASE_NS, PHASE_EW]

ACTION_KEEP = 0
ACTION_SWITCH = 1
N_ACTIONS = 2

LANES = ["N", "S", "E", "W"]


class TrafficEnv:
    def __init__(
        self,
        max_steps=1000,
        arrival_rates=(0.10, 0.10, 0.25, 0.25),
        departure_rate=1.0,
        yellow_time=3,
        min_green=5,
        max_queue=40,
        seed=None,
    ):
        self.max_steps = max_steps
        self.arrival_rates = np.array(arrival_rates, dtype=np.float32)
        self.departure_rate = departure_rate
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_queue = max_queue
        self.rng = np.random.default_rng(seed)

        self.state_dim = 6
        self.n_actions = N_ACTIONS

        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.queues = np.zeros(4, dtype=np.int32)
        self.waiting = np.zeros(4, dtype=np.float32)
        self.phase = PHASE_NS
        self.time_in_phase = 0
        self.yellow_remaining = 0
        self.step_count = 0
        self.prev_total_wait = 0.0
        self.history = {"wait": [], "queue": [], "throughput": [], "max_queue_step": []}
        self.cars_passed = 0
        self.max_queue_seen = 0
        return self._get_state()

    def _get_state(self):
        norm_q = np.clip(self.queues / self.max_queue, 0, 1).astype(np.float32)
        return np.array(
            [
                norm_q[0], norm_q[1], norm_q[2], norm_q[3],
                float(self.phase),
                min(self.time_in_phase / 60.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _green_lanes(self):
        if self.phase == PHASE_NS:
            return [0, 1]
        return [2, 3]

    def step(self, action):
        if self.yellow_remaining > 0:
            self.yellow_remaining -= 1
        elif action == ACTION_SWITCH and self.time_in_phase >= self.min_green:
            self.yellow_remaining = self.yellow_time
            self.phase = 1 - self.phase
            self.time_in_phase = 0

        arrivals = self.rng.poisson(self.arrival_rates)
        self.queues = np.minimum(self.queues + arrivals, self.max_queue)

        passed = 0
        if self.yellow_remaining == 0:
            for lane in self._green_lanes():
                if self.queues[lane] > 0:
                    leaving = min(self.queues[lane], int(self.departure_rate))
                    self.queues[lane] -= leaving
                    passed += leaving

        self.waiting = np.where(self.queues > 0, self.waiting + 1.0, 0.0)
        self.cars_passed += passed
        self.time_in_phase += 1
        self.step_count += 1

        total_wait = float(np.sum(self.queues * self.waiting))
        reward = -float(np.sum(self.queues)) - 0.1 * float(np.sum(self.waiting))

        step_max_q = int(np.max(self.queues))
        self.max_queue_seen = max(self.max_queue_seen, step_max_q)
        self.history["wait"].append(float(np.sum(self.waiting)))
        self.history["queue"].append(int(np.sum(self.queues)))
        self.history["throughput"].append(passed)
        self.history["max_queue_step"].append(step_max_q)
        self.prev_total_wait = total_wait

        done = self.step_count >= self.max_steps
        info = {
            "queues": self.queues.copy(),
            "waiting": self.waiting.copy(),
            "phase": self.phase,
            "yellow": self.yellow_remaining > 0,
            "cars_passed": self.cars_passed,
            "throughput": passed,
        }
        return self._get_state(), reward, done, info

    def metrics(self):
        return {
            "avg_queue": float(np.mean(self.history["queue"])) if self.history["queue"] else 0.0,
            "avg_wait_per_step": float(np.mean(self.history["wait"])) if self.history["wait"] else 0.0,
            "total_throughput": int(self.cars_passed),
            "max_queue": int(self.max_queue_seen),
        }
