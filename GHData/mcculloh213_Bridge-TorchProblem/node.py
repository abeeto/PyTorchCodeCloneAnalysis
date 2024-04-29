import sys
import puzzlestate as ps

__author__ = "H.D. 'Chip' McCullough IV"


class Node:
    def __init__(self, state, parent, action, cost, heuristic=0, depth=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = cost
        self.h = heuristic
        self.depth = depth

    def get_state(self):
        return self.state

    def get_parent(self):
        return self.parent

    def get_action(self):
        return self.action

    def get_path_cost(self):
        return self.g

    def get_heuristic_cost(self):
        return self.h

    def get_depth(self):
        return self.depth

    def get_f_cost(self):
        return self.g + self.h

    def get_child_nodes_no_h(self, statemap, rulemap):
        ns = statemap[self.state]
        ac = rulemap[self.state]
        children = []
        for i in range(0, len(ns)):
            children.append(Node(ns[i], self, ac[i][0], self.g + ac[i][1], 0, self.depth+1))
        return children

    def get_child_nodes_h(self, statemap, rulemap):
        ns = statemap[self.state]
        ac = rulemap[self.state]
        children = []
        for i in range(0, len(ns)):
            children.append(Node(ns[i], self, ac[i][0], self.g + ac[i][1], ac[i][2], self.depth+1))
        return children

    def __lt__(self, other):
        return self.g + self.h < other.get_f_cost()

    def __eq__(self, other):
        return self.state == other.get_state()


class PuzzleNode:
    def __init__(self, state, parent, action, depth):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = 1
        self.h = state.get_heuristic()
        self.depth = depth

    def get_state(self):
        return self.state.get_state()

    def get_class(self):
        return self.state

    def get_parent(self):
        return self.parent

    def get_action(self):
        return self.action

    def get_path_cost(self):
        return self.g

    def get_heuristic_cost(self):
        return self.h

    def get_depth(self):
        return self.depth

    def get_f_cost(self):
        return self.g + self.h

    def get_child_nodes_h(self):
        ms = self.state.move_set()
        children = []
        for m in ms:
            if m == "MOVE LEFT":
                s = self.state.move_left()
                pz = ps.PuzzleState(s[0], s[1], s[2])
                children.append(PuzzleNode(pz, self, "MOVE LEFT", self.depth+1))
            elif m == "MOVE RIGHT":
                s = self.state.move_right()
                pz = ps.PuzzleState(s[0], s[1], s[2])
                children.append(PuzzleNode(pz, self, "MOVE RIGHT", self.depth + 1))
            elif m == "MOVE UP":
                s = self.state.move_up()
                pz = ps.PuzzleState(s[0], s[1], s[2])
                children.append(PuzzleNode(pz, self, "MOVE UP", self.depth + 1))
            elif m == "MOVE DOWN":
                s = self.state.move_down()
                pz = ps.PuzzleState(s[0], s[1], s[2])
                children.append(PuzzleNode(pz, self, "MOVE DOWN", self.depth + 1))
            else:
                print("ERROR: Unexpected Input.")
                sys.exit(13)

        return children

    # __lt__ actually is now __gt__ in order to use heapq as a max heap
    def __lt__(self, other):
        return self.g + self.h > other.get_f_cost()

    def __eq__(self, other):
        return self.state.get_state() == other.get_state()

