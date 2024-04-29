#!/usr/bin/python

import lightsource


class Adventure:
    def __init__(self):
        self.turn_count = 0
        self.adventurers = []
        self.lightsources = []

    def add_adventurer(self, adventurer):
        if self.adventurers.count(adventurer) == 0:
            self.adventurers.append(adventurer)
            return True
        else:
            return False

    def add_lightsource(self, lightsource):
        self.lightsources.append(lightsource)

    def set_current_turn(self, turn):
        self.turn_count = turn

    def tick(self):
        self.turn_count += 1
        for source in self.lightsources:
            source.decrement_time_remaining()
        if self.turn_count % 4 == 0:
            print "Grind!"

    def get_adventurer_list(self):
        return self.adventurers
