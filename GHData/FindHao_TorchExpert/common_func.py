import glob
import os


def merge_interval(intervals):
    """
    Merge intervals that are overlapped.
    Args:
        intervals: a list of intervals
    Returns:
        a list of merged intervals
    """
    if not intervals:
        return []
    intervals.sort(key=lambda x: x.start_time_ns)
    res = []
    res.append(intervals[0])
    for i in range(1, len(intervals)):
        last = res[-1]
        if intervals[i].start_time_ns <= last.end_time_ns:
            last.end_time_ns = max(last.end_time_ns, intervals[i].end_time_ns)
            last.include_events.extend(intervals[i].include_events)
        else:
            res.append(intervals[i])
    return res

def get_latest_file(path):
    """
    Get the latest file in a directory.
    Args:
        path: the directory path
    Returns:
        the latest file path
    """
    list_of_files = glob.glob(path + '/*')
    list_of_files = [ _ for _ in list_of_files if os.path.isfile(_) and _.endswith(".json")]
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file