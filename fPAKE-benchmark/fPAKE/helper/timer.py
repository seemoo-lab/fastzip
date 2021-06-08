import time

class timer:
    """
    calculates time in ns
    """
    def __init__(self):
        self.start = time.perf_counter_ns()
    def start_time(self):
        self.start = time.perf_counter_ns()
    def stop_time(self):
        end = time.perf_counter_ns()
        return int(end - self.start)
