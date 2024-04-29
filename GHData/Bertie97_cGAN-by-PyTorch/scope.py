
import time

class timer(object):
    def __init__(self, name='', off=None):
        if off: name = ''
        self.name = name
        self.nround = 0
    def __enter__(self):
        self.start = time.time()
        self.prevtime = self.start
        return self
    def round(self, name = ''):
        self.nround += 1
        self.end = time.time()
        if self.name:
            if not name: name = "%s(round%d)"%(self.name, self.nround)
            print("[%s takes %lfs]"%(name, self.end - self.prevtime))
        self.prevtime = self.end
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type == RuntimeError and str(exc_value) == "JUMP": return True
        if self.name:
            print("[%s%s takes %lfs]"%
                  (self.name, '' if self.nround == 0 else "(all)", time.time() - self.start))
            
class JUMP(object):
    def __init__(self, jump=None): self.jump = True if jump is None else jump
    def __enter__(self):
        def dojump(): raise RuntimeError("JUMP")
        if self.jump: dojump()
        else: return dojump
    def __exit__(self, *args): pass
    def __call__(self, condition): return JUMP(condition)
    
def scope(name, timing=True):
    return timer(name, not timing)
jump = JUMP()