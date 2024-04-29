import sys

__author__ = "H.D. 'Chip' McCullough IV"

# top: 5 4 0  ->  1 2 3
# mid: 6 1 8  ->  8 0 4
# low: 7 3 2  ->  7 6 5


class PuzzleState:
    def __init__(self, top, mid, low):
        self.top = top
        self.mid = mid
        self.low = low
        self.h = self.heuristic()

    def get_state(self):
        return [self.top, self.mid, self.low]

    def get_heuristic(self):
        return self.h

    def where_is_the_zero(self):
        puzzle = [False, False, False]
        if '0' in self.top:
            puzzle[0] = True
        elif '0' in self.mid:
            puzzle[1] = True
        else:
            puzzle[2] = True

        return puzzle

    def move_set(self):
        zero = self.where_is_the_zero()
        moveset = []

        if zero[0]:  # Zero is in top
            if self.top.index('0') == 0:
                moveset.append("MOVE LEFT")
                moveset.append("MOVE UP")
            elif self.top.index('0') == 1:
                moveset.append("MOVE RIGHT")
                moveset.append("MOVE UP")
                moveset.append("MOVE LEFT")
            else:
                moveset.append("MOVE RIGHT")
                moveset.append("MOVE UP")
        elif zero[1]:  # Zero is in mid
            if self.mid.index('0') == 0:
                moveset.append("MOVE DOWN")
                moveset.append("MOVE LEFT")
                moveset.append("MOVE UP")
            elif self.mid.index('0') == 1:
                moveset.append("MOVE DOWN")
                moveset.append("MOVE RIGHT")
                moveset.append("MOVE LEFT")
                moveset.append("MOVE UP")
            else:
                moveset.append("MOVE DOWN")
                moveset.append("MOVE RIGHT")
                moveset.append("MOVE UP")
        else:  # Zero is in low
            if self.low.index('0') == 0:
                moveset.append("MOVE DOWN")
                moveset.append("MOVE LEFT")
            elif self.low.index('0') == 1:
                moveset.append("MOVE DOWN")
                moveset.append("MOVE RIGHT")
                moveset.append("MOVE LEFT")
            else:
                moveset.append("MOVE DOWN")
                moveset.append("MOVE RIGHT")

        return moveset

    def move_left(self):
        top = [-1, -1, -1]
        mid = [-1, -1, -1]
        low = [-1, -1, -1]
        zero = self.where_is_the_zero()

        if zero[0]:  # Zero is in top
            if self.top.index('0') == 0:  # Swap top[0] and top[1]
                top[0] = self.top[1]
                top[1] = self.top[0]
                top[2] = self.top[2]
                mid = self.mid
                low = self.low
            elif self.top.index('0') == 1:  # Swap top[1] and top[2]
                top[0] = self.top[0]
                top[1] = self.top[2]
                top[2] = self.top[1]
                mid = self.mid
                low = self.low
            else:
                print("ERROR: Unexpected Input.")
                sys.exit(1)
        elif zero[1]:  # Zero is in mid
            if self.mid.index('0') == 0:  # Swap mid[0] and mid[1]
                top = self.top
                mid[0] = self.mid[1]
                mid[1] = self.mid[0]
                mid[2] = self.mid[2]
                low = self.low
            elif self.mid.index('0') == 1:  # Swap mid[1] and mid[2]
                top = self.top
                mid[0] = self.mid[0]
                mid[1] = self.mid[2]
                mid[2] = self.mid[1]
                low = self.low
            else:
                print("ERROR: Unexpected Input.")
                sys.exit(2)
        else:  # Zero is in low
            if self.low.index('0') == 0:  # Swap low[0] and low[1]
                top = self.top
                mid = self.mid
                low[0] = self.low[1]
                low[1] = self.low[0]
                low[2] = self.low[2]
            elif self.low.index('0') == 1:  # Swap loq[1] and low[0]
                top = self.top
                mid = self.mid
                low[0] = self.low[0]
                low[1] = self.low[2]
                low[2] = self.low[1]
            else:
                print("ERROR: Unexpected Input.")
                sys.exit(3)

        return [top, mid, low]

    def move_right(self):
        top = [-1, -1, -1]
        mid = [-1, -1, -1]
        low = [-1, -1, -1]
        zero = self.where_is_the_zero()

        if zero[0]:  # Zero is in top
            if self.top.index('0') == 0:
                print("ERROR: Unexpected Input.")
                sys.exit(4)
            elif self.top.index('0') == 1:  # Swap top[0] and top[1]
                top[0] = self.top[1]
                top[1] = self.top[0]
                top[2] = self.top[2]
                mid = self.mid
                low = self.low
            else:  # Swap top[1] and top[2]
                top[0] = self.top[0]
                top[1] = self.top[2]
                top[2] = self.top[1]
                mid = self.mid
                low = self.low
        elif zero[1]:  # Zero is in mid
            if self.mid.index('0') == 0:
                print("ERROR: Unexpected Input.")
                sys.exit(5)
            elif self.mid.index('0') == 1:  # Swap mid[0] and mid[1]
                top = self.top
                mid[0] = self.mid[1]
                mid[1] = self.mid[0]
                mid[2] = self.mid[2]
                low = self.low
            else:  # Swap mid[1] and mid[2]
                top = self.top
                mid[0] = self.mid[0]
                mid[1] = self.mid[2]
                mid[2] = self.mid[1]
                low = self.low
        else:  # Zero is in low
            if self.low.index('0') == 0:
                print("ERROR: Unexpected Input.")
                sys.exit(6)
            elif self.low.index('0') == 1:  # Swap low[0] and low[1]
                top = self.top
                mid = self.mid
                low[0] = self.low[1]
                low[1] = self.low[0]
                low[2] = self.low[2]
            else:  # Swap loq[1] and low[0]
                top = self.top
                mid = self.mid
                low[0] = self.low[0]
                low[1] = self.low[2]
                low[2] = self.low[1]

        return [top, mid, low]

    def move_up(self):
        top = [-1, -1, -1]
        mid = [-1, -1, -1]
        low = [-1, -1, -1]
        zero = self.where_is_the_zero()

        if zero[0]:  # Zero is in top
            if self.top.index('0') == 0:  # Swap top[0] and mid[0]
                top[0] = self.mid[0]
                top[1] = self.top[1]
                top[2] = self.top[2]
                mid[0] = self.top[0]
                mid[1] = self.mid[1]
                mid[2] = self.mid[2]
                low = self.low
            elif self.top.index('0') == 1:  # Swap top[1] and mid[1]
                top[0] = self.top[0]
                top[1] = self.mid[1]
                top[2] = self.top[2]
                mid[0] = self.mid[0]
                mid[1] = self.top[1]
                mid[2] = self.mid[2]
                low = self.low
            else:  # Swap top[2] and mid[2]
                top[0] = self.top[0]
                top[1] = self.top[1]
                top[2] = self.mid[2]
                mid[0] = self.mid[0]
                mid[1] = self.mid[1]
                mid[2] = self.top[2]
                low = self.low
        elif zero[1]:  # Zero is in mid
            if self.mid.index('0') == 0:  # Swap mid[0] and low[0]
                top = self.top
                mid[0] = self.low[0]
                mid[1] = self.mid[1]
                mid[2] = self.mid[2]
                low[0] = self.mid[0]
                low[1] = self.low[1]
                low[2] = self.low[2]
            elif self.mid.index('0') == 1:  # Swap mid[1] and low[1]
                top = self.top
                mid[0] = self.mid[0]
                mid[1] = self.low[1]
                mid[2] = self.mid[2]
                low[0] = self.low[0]
                low[1] = self.mid[1]
                low[2] = self.low[2]
            else:  # Swap mid[2] and low[2]
                top = self.top
                mid[0] = self.mid[0]
                mid[1] = self.mid[1]
                mid[2] = self.low[2]
                low[0] = self.low[0]
                low[1] = self.low[1]
                low[2] = self.mid[2]
        else:  # Zero is in low
            if self.low.index('0') == 0:
                print("ERROR: Unexpected Input.")
                sys.exit(7)
            elif self.low.index('0') == 1:
                print("ERROR: Unexpected Input.")
                sys.exit(8)
            else:
                print("ERROR: Unexpected Input.")
                sys.exit(9)

        return [top, mid, low]

    def move_down(self):
        top = [-1, -1, -1]
        mid = [-1, -1, -1]
        low = [-1, -1, -1]
        zero = self.where_is_the_zero()

        if zero[0]:  # Zero is in top
            if self.top.index('0') == 0:
                print("ERROR: Unexpected Input.")
                sys.exit(10)
            elif self.top.index('0') == 1:
                print("ERROR: Unexpected Input.")
                sys.exit(11)
            else:
                print("ERROR: Unexpected Input.")
                sys.exit(12)
        elif zero[1]:  # Zero is in mid
            if self.mid.index('0') == 0:  # Swap top[0] and mid[0]
                top[0] = self.mid[0]
                top[1] = self.top[1]
                top[2] = self.top[2]
                mid[0] = self.top[0]
                mid[1] = self.mid[1]
                mid[2] = self.mid[2]
                low = self.low
            elif self.mid.index('0') == 1:  # Swap top[1] and mid[1]
                top[0] = self.top[0]
                top[1] = self.mid[1]
                top[2] = self.top[2]
                mid[0] = self.mid[0]
                mid[1] = self.top[1]
                mid[2] = self.mid[2]
                low = self.low
            else:  # Swap top[2] and mid[2]
                top[0] = self.top[0]
                top[1] = self.top[1]
                top[2] = self.mid[2]
                mid[0] = self.mid[0]
                mid[1] = self.mid[1]
                mid[2] = self.top[2]
                low = self.low
        else:  # Zero is in low
            if self.low.index('0') == 0:  # Swap mid[0] and low[0]
                top = self.top
                mid[0] = self.low[0]
                mid[1] = self.mid[1]
                mid[2] = self.mid[2]
                low[0] = self.mid[0]
                low[1] = self.low[1]
                low[2] = self.low[2]
            elif self.low.index('0') == 1:  # Swap mid[1] and low[1]
                top = self.top
                mid[0] = self.mid[0]
                mid[1] = self.low[1]
                mid[2] = self.mid[2]
                low[0] = self.low[0]
                low[1] = self.mid[1]
                low[2] = self.low[2]
            else:  # Swap mid[2] and low[2]
                top = self.top
                mid[0] = self.mid[0]
                mid[1] = self.mid[1]
                mid[2] = self.low[2]
                low[0] = self.low[0]
                low[1] = self.low[1]
                low[2] = self.mid[2]
        return [top, mid, low]

    def heuristic(self):
        h = 0

        if self.top[0] == '1':
            h += 1
        if self.top[1] == '2':
            h += 1
        if self.top[2] == '3':
            h += 1
        if self.mid[0] == '8':
            h += 1
        if self.mid[1] == '0':
            h += 1
        if self.mid[2] == '4':
            h += 1
        if self.low[0] == '7':
            h += 1
        if self.low[1] == '6':
            h += 1
        if self.low[2] == '5':
            h += 1

        return h

    def print_state(self):
        print("-------")
        print("|{0}|{1}|{2}|".format(self.top[0], self.top[1], self.top[2]))
        print("|{0}|{1}|{2}|".format(self.mid[0], self.mid[1], self.mid[2]))
        print("|{0}|{1}|{2}|".format(self.low[0], self.low[1], self.low[2]))
        print("-------")
