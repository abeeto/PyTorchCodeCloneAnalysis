import indirect


class base:
    def foo(self):
        return 3


class test(base):
    def __init__(self):
        pass

    def fun(self):
        self.foo()
        super().foo()