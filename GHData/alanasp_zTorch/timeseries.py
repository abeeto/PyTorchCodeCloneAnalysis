import numpy as np
import utils

class TSEntry:
    def __init__(self, entry, time):
        # entry should be np.array
        self.entry = entry
        self.time = time
        self.next = None
        self.prev = None


class TimeSeries:
    # if file_prefix provided, loads data from file file_prefixT upon request for time series at time T
    # max_time only needed with file_prefix
    def __init__(self, init_entry, file_prefix=None, on_the_fly=False, std=None, max_time=None):
        self.file_prefix = file_prefix
        self.on_the_fly = on_the_fly
        self.std = std
        self.max_time = max_time
        # disk mode
        if file_prefix is not None and not on_the_fly:
            if max_time is None:
                raise Exception('Maximum time for time series not specified!')
            self.current = TSEntry(init_entry, 0)
            # first/last references not supported when loading from file
            self.first = None
            self.last = None

        # memory mode
        elif not on_the_fly:
            self.current = TSEntry(init_entry, 0)
            self.first = self.current
            self.last = self.first

        # on the fly mode: data generated as needed and discarded
        elif on_the_fly:
            if std is None or max_time is None:
                raise Exception('Have to provide std and max_time with on_the_fly option!')
            self.current = TSEntry(init_entry, 0)
            self.first = self.current
            self.last = self.first

        else:
            raise Exception('Invalid options!')

        self.shape = init_entry.shape
        self.time_elapsed = 0

    def add_entry_delta(self, delta):
        if self.on_the_fly:
            raise Exception('Cannot add entries in on_the_fly mode!')
        new_entry = TSEntry(utils.cap_profiles(self.current.entry+delta), self.time_elapsed+1)
        self.add_entry(new_entry)

    def add_entry(self, entry):
        if self.on_the_fly:
            raise Exception('Cannot add entries in on_the_fly mode!')
        if self.file_prefix:
            self.current = entry
            self.time_elapsed = entry.time
            with open(self.file_prefix + str(entry.time), 'wb') as data_file:
                data_file.write(entry.entry.tostring())
        else:
            entry.prev = self.last
            self.last.next = entry
            self.last = entry
            self.current = entry
            self.time_elapsed += 1

    def load_entry(self, delta=None, time=None):
        if self.file_prefix is None:
            raise Exception('Entry loading is only for disk-based time series!')
        if delta is not None:
            # could cache next delta file here
            time = self.time_elapsed + delta
        if time is None:
            raise Exception('Specify time or delta!')
        if time > self.max_time:
            self.current = None
        else:
            with open(self.file_prefix + str(time), 'rb') as data_file:
                if time == 0:
                    data_file.readline()
                    data_file.readline()
                ts_entry = TSEntry(np.reshape(np.fromstring(data_file.read()), (-1, 3)), time)
                self.current = ts_entry
                self.time_elapsed = time
                self.shape = ts_entry.entry.shape

    def get_next(self, delta=1):
        if self.on_the_fly:
            if self.time_elapsed + delta > self.max_time:
                self.current = None
            else:
                profile_delta = np.random.normal(0, self.std, self.shape)
                new_entry = TSEntry(self.first.entry + profile_delta, self.time_elapsed + delta)
                self.current = new_entry
                self.time_elapsed += delta
        elif self.file_prefix is not None:
            self.load_entry(delta=delta)
        else:
            count = 0
            while count < delta:
                self.current = self.current.next
        return self.current
