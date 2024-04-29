import numpy as np
from scipy.stats import norm


# only retains the latest elements added up to set size
class SlidingWindow:
    def __init__(self, size):
        self.data = list()
        self.latest = 0
        self.size = size

    def __len__(self):
        return len(self.data)

    # 0th index represents item most recently added
    def __getitem__(self, item):
        if len(self) > 0:
            return self.data[(self.latest + item) % len(self.data)]
        return None

    def append(self, item):
        if len(self.data) < self.size:
            self.data.append(item)
            self.latest = len(self.data) - 1
        else:
            self.latest = (self.latest + 1) % self.size
            self.data[self.latest] = item

    def get_latest(self):
        if len(self) > 0:
            return self.data[self.latest]
        return None


class DecisionEngine:
    # units in seconds
    mon_periods = [2, 5, 10, 20, 50]
    window_size = 500

    # thresholds indicate danger levels which we don't want to miss
    # user has to specify at least one of the thresholds
    # 0 < confidence < 1
    def __init__(self, name, init_period, lower_threshold=None, upper_threshold=None, confidence=0.95):
        if lower_threshold is None and upper_threshold is None:
            raise Exception('Specify at least one threshold!')
        self.name = name
        self.change_window = SlidingWindow(self.window_size)
        self.val_window = SlidingWindow(self.window_size)
        self.curr_period = init_period

        if lower_threshold is None:
            lower_threshold = -1e10
        if upper_threshold is None:
            upper_threshold = 1e10
        self.ok_interval = (lower_threshold, upper_threshold)
        self.confidence = confidence

    def feed_data(self, value, timestamp=None):
        # assume value stayed the same throughout the period
        change = 0
        if len(self.val_window) > 0:
            change = value - self.val_window[0]
        # all forecasts are based on change rather than the values themselves
        for i in range(self.curr_period):
            self.change_window.append(change)
            self.val_window.append(value)

    def get_decision(self):
        # keep default period until we collect more data points
        if len(self.val_window) < self.window_size:
            return None
        np_window = np.array(self.change_window.data)
        curr_val = self.val_window[0]
        change_mean = np.mean(np_window)
        change_var = np.var(np_window)

        # pick largest period which doesn't cross thresholds with 'confidence' probability
        for period in reversed(self.mon_periods):
            rnd_walk_std = np.sqrt(change_var*period)
            rnd_walk_std = max(0.01*period, rnd_walk_std)
            change_interval = norm.interval(self.confidence, loc=change_mean, scale=rnd_walk_std)
            if curr_val + change_interval[0] > self.ok_interval[0] and \
               curr_val + change_interval[1] < self.ok_interval[1]:
                # period stays unchanged, so no decision to change
                if self.curr_period == period:
                    return None
                self.curr_period = period
                return period
        if self.curr_period == self.mon_periods[0]:
            return None
        return self.mon_periods[0]


#de = DecisionEngine('aaa', 10, 0, 100, 0.9)
#for i in range(50):
#    val = np.random.normal(80+i/10, 5)
#    de.feed_data(50)

#for i in range(900):
#    val = np.random.normal(85 - i / 10, 5)
#    de.feed_data(50)
#    d = de.get_decision()
#    if d is None:
#        print('None')
#    else:
#        print(d)
