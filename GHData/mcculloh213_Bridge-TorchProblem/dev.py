from node import Node
from state import state
from action import rules
from AStarAgent import astar, greedysearch
from UCAgent import ucs
from ParseInput import parseinput

__author__ = "H.D. 'Chip' McCullough IV"  # FOR DEVELOPMENT PURPOSES ONLY

s = (False, False, False, False, False)
n = Node(s, None, "START", 0)
ns = n.get_child_nodes_no_h(state, rules)

ucs(s, (True, True, True, True, True))

# print(parseinput("ADP", "CD"))

t = "5 4 0".split()
m = "6 1 8".split()
l = "7 3 2".split()
te = "1 2 3".split()
me = "8 0 4".split()
le = "7 6 5".split()

pz = greedysearch([t, m, l], [te, me, le])
