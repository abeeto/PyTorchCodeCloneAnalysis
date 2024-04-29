import heapq
from action import rules
from node import Node, PuzzleNode
from state import state
from puzzlestate import PuzzleState as ps

__author__ = "H.D. 'Chip' McCullough IV"

#  A* Search: f(n) = g(n) + h(n)


def astar(start, end):
    cur = Node(start, None, "START", 0)  # Initial state has no parent
    fringe = [cur]
    expanded = set()
    soln = False
    while (len(fringe) > 0) and not soln:
        node = heapq.heappop(fringe)
        if node.get_state() == end:
            soln = node  # soln is no longer false -- break
        expanded.add(node.get_state())
        children = node.get_child_nodes_h(state, rules)
        for c in children:
            if c.get_state() not in expanded or (c not in fringe and c.get_action() != "NoOp"):
                    heapq.heappush(fringe, c)
    return soln


def greedysearch(start, end):
    s = ps(start[0], start[1], start[2])
    cur = PuzzleNode(s, None, "START", 0)
    fringe = [cur]
    expanded = []
    soln = False
    while (len(fringe) > 0) and not soln:
        node = heapq.heappop(fringe)
        if node.get_state() == end:
            soln = node
        expanded.append(node.get_state())
        children = node.get_child_nodes_h()
        for c in children:
            if c.get_state() not in expanded or c not in fringe:
                heapq.heappush(fringe, c)

    return soln
