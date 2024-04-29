import torch
import traceback


class cutotime():
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        torch.cuda.synchronize()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        self.start.record()
        torch.cuda.synchronize()

    def __exit__(self, exc_type, exc_value, tb):
        torch.cuda.synchronize()
        self.end.record()
        torch.cuda.synchronize()
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
        print('{} - {:.5f} ms'.format(self.name, self.start.elapsed_time(self.end)))

    def start(self):
        self.__enter__()
        return self

    def stop(self):
        self.__exit__(None, None, None)



