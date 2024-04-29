class ProfileEventSlim:
    """
    A simplified version of event class. 
    Attributes:
        duration_us: duration of the event in microseconds(us)
        start_us: start time of the event in microseconds(us)
        end_us: end time of the event in microseconds(us)
        include_events: a list of raw events that have overlaps
    """
    def __init__(self, event=None, duration_time_ns=None, start_time_ns=None, end_time_ns=None):
        if event is not None:
            # self.duration_time_ns = event.duration_time_ns
            # self.start_time_ns = event.start_time_ns
            # self.end_time_ns = event.end_time_ns
            self.duration_time_ns = event.duration_us() * 1e3
            self.start_time_ns = event.start_us() * 1e3
            self.end_time_ns = (event.start_us() + event.duration_us()) * 1e3
            self.include_events = [event]
        else:
            self.duration_time_ns = duration_time_ns
            self.start_time_ns = start_time_ns
            self.end_time_ns = end_time_ns
            self.include_events = []


class TraceEvent:
    """
    A class to store the trace event.
    """
    def __init__(self, tmp_dict=None) -> None:
        if tmp_dict is not None:
            self.name = tmp_dict["name"]
            self.ph = tmp_dict["ph"]
            self.pid = tmp_dict["pid"]
            self.tid = tmp_dict["tid"]
            self.start_time = tmp_dict["ts"]
            self.duration = tmp_dict["dur"]
            self.args = tmp_dict["args"]
        else:
            self.name = None
            self.ph = None
            self.pid = None
            self.tid = None
            self.start_time = None
            self.duration = None
            self.args = None
            