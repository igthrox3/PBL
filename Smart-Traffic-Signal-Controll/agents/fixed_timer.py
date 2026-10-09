from traffic_env import ACTION_KEEP, ACTION_SWITCH


class FixedTimerAgent:
    def __init__(self, period=20):
        self.period = period

    def act(self, state, greedy=True):
        time_in_phase = state[5] * 60.0
        if time_in_phase >= self.period:
            return ACTION_SWITCH
        return ACTION_KEEP

    def update(self, *args, **kwargs):
        pass
