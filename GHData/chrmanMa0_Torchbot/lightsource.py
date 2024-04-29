#!/usr/bin/python


def main():
    torch = LightSource("torch", 2, 2, 2)
    candle = LightSource("candle", 4, 1, 1)
    cellos = Adventurer("cellos")
    fenring = Adventurer("fenring")
    stephen = Adventurer("stephen")
    torch.add_bright_light(cellos)
    torch.add_bright_light(fenring)
    torch.add_dim_light(stephen)
    candle.add_bright_light(cellos)
    candle.add_dim_light(stephen)
    cellos.print_bright_count()
    fenring.print_bright_count()
    stephen.print_bright_count()
    cellos.print_dim_count()
    fenring.print_dim_count()
    stephen.print_dim_count()

    torch.print_bright()
    torch.print_dim()
    cellos.print_current_lighting()
    fenring.print_current_lighting()
    stephen.print_current_lighting()

    torch.decrement_time_remaining()
    torch.print_time_remaining()
    torch.decrement_time_remaining()
    torch.print_time_remaining()
    torch.print_bright()
    fenring.print_current_lighting()
    cellos.print_current_lighting()
    stephen.print_current_lighting()


class Adventurer:
    def __init__(self, name):
        self.name = name
        self.bright_sources = 0
        self.dim_sources = 0
        self.torches_available = 0
        self.candles_available = 0
        self.lantern_available = False
        self.lantern_oil_available = 0
    def add_torches(self, torches):
        self.torches_available += torches

    def add_candles(self, candles):
        self.candles_available += candles

    def add_lantern(self):
        self.lantern_available = True

    def add_lantern_oil(self, oil_count):
        self.lantern_oil_available += oil_count

    def use_torch(self):
        if self.torches_available > 0:
            self.torches_available -= 1
            return True
        else:
            return False

    def use_candle(self):
        if self.candles_available > 0:
            self.candles_available -= 1
            return True
        else:
            return False

    def use_lantern(self):
        if self.lantern_available and self.lantern_oil_available > 0:
            self.lantern_oil_available -= 1
            return True
        else:
            return False

    def print_current_lighting(self):
        print self.name + " is in " + self.calculate_lighting()
    def get_torch_count(self):
        return self.torches_available

    def calculate_lighting(self):
        if self.bright_sources > 0:
            return "bright"
        elif self.dim_sources > 0:
            return "dim"
        else:
            return "darkness"

    def add_bright(self):
        self.bright_sources = self.bright_sources + 1

    def add_dim(self):
        self.dim_sources = self.dim_sources + 1

    def remove_bright(self):
        self.bright_sources = self.bright_sources - 1

    def remove_dim(self):
        self.dim_sources = self.dim_sources - 1

    def print_bright_count(self):
        print self.name + " has " + str(self.bright_sources) + " bright sources"

    def print_dim_count(self):
        print self.name + " has " + str(self.dim_sources) + " dim sources"

    def get_name(self):
        return self.name


class LightSource:
    def __init__(self, type_name, time_remaining, bright_max, dim_max):
        self.time_remaining_ = time_remaining
        self.bright_max_ = bright_max
        self.dim_max_ = dim_max
        self.bright_lighting_list_ = []
        self.dim_lighting_list_ = []
        self.type_name = type_name

    def get_type(self):
        return self.type_name

    def add_bright_light(self, recipient):
        print "list length " + str(len(self.bright_lighting_list_)) + " and max " + str(self.bright_max_)
        if len(self.bright_lighting_list_) <= self.bright_max_:
            print recipient.name + " is getting bright light"
            self.bright_lighting_list_.append(recipient)
            recipient.add_bright()
            return True
        else:
            return False

    def add_dim_light(self, recipient):
        if len(self.dim_lighting_list_) <= self.dim_max_:
            self.dim_lighting_list_.append(recipient)
            recipient.add_dim()
            return True
        else:
            return False

    def replace_bright_light(self, new_bright, old_bright):
        self.bright_lighting_list_.remove(old_bright)
        self.bright_lighting_list_.append(new_bright)

    def replace_dim_light(self, new_dim, old_dim):
        self.dim_lighting_list_.remove(old_dim)
        self.dim_lighting_list_.append(new_dim)

    def decrement_time_remaining(self):
        if self.time_remaining_ > 0:
            self.time_remaining_ = self.time_remaining_ - 1
            if self.time_remaining_ == 0:
                for recipient in self.bright_lighting_list_:
                    print "Removing ", recipient.name
                    recipient.remove_bright()
                del self.bright_lighting_list_[0:len(self.bright_lighting_list_)]
                for recipient in self.dim_lighting_list_:
                    recipient.remove_dim()
                del self.dim_lighting_list_[0:len(self.dim_lighting_list_)]

    def print_bright(self):
        for recipient in self.bright_lighting_list_:
            print recipient.name

    def print_dim(self):
        for recipient in self.dim_lighting_list_:
            print recipient.name

    def print_time_remaining(self):
        print self.time_remaining_

if __name__ == "__main__":
    main()
