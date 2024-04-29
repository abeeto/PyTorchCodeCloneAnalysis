import torch

class TorchTimer:
    def __init__(self, name="Timed Event"):
        self.s = torch.cuda.Event(enable_timing=True)
        self.e = torch.cuda.Event(enable_timing=True)
        self.name = name

    def start(self):
        self.s.record()

    def end(self):
        self.e.record()
        torch.cuda.synchronize()
        return self.s.elapsed_time(self.e)

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        print(f"{self.name:<20} {self.end():9.3f}")
