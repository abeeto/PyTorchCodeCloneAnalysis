class ToTrain:
    def __init__(self):
        pass

    def next(self, trainer):
        raise NotImplementedError("Not implemented!")


class TwoFiveRule(ToTrain):
    """Implementation of ToTrain which follows a simple 2-5 ratio rule: Train G for 2 epochs, and D for 5."""

    def __init__(self):
        ToTrain.__init__(self)
        self.state = 0

    def next(self, trainer):
        if self.state < 2:
            self.state += 1
            return "G"
        else:
            self.state += 1
            if self.state >= 7:
                self.state = 0
            return "D"


class AlwaysDRule(ToTrain):
    """Implementation of ToTrain which always trains the discriminator (for use with dynamic training)"""

    def __init__(self):
        ToTrain.__init__(self)

    def next(self, trainer):
        return "D"


class AlwaysGRule(ToTrain):
    """Implementation of ToTrain which always trains the generator"""

    def __init__(self):
        ToTrain.__init__(self)

    def next(self, trainer):
        return "G"
