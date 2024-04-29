import heapq
from action import rules
from node import Node
from state import state

__author__ = "H.D. 'Chip' McCullough IV"


#  Uniform Cost Search: f(n) = g(n)
#  PSEUDOCODE -- Artificial Intelligence: A Modern Approach (Russell, Norvig)
#  FUNCTION UniformCostSearch(problem) RETURNS [a solution, or failure]
#      node <- Node(State = problem.initial_state, g(n) = 0)
#      fringe <- PriorityQueue<o=g(n)> = {node}
#      expanded <- Set = {}
#      LOOP DO
#          IF fringe.empty? THEN RETURN False
#          node <- fringe.pop
#          IF problem.GOAL_TEST(node.State) THEN RETURN Solution(node)
#          expanded.add(node)
#          FOR EACH action in problem.Actions(node.State) DO
#              child <- Child_Node(problem, node, action)
#              IF child.State NOT IN expanded | child.State NOT IN fringe THEN
#                  fringe.insert(child)
#              ELSE IF child.State in fringe [with higher g(n)] THEN
#                  node = child


def ucs(start, end):
    cur = Node(start, None, "START", 0)  # Initial state has no parent
    fringe = [cur]
    expanded = set()
    soln = False
    while (len(fringe) > 0) and not soln:
        node = heapq.heappop(fringe)
        if node.get_state() == end:
            soln = node  # soln is no longer false -- break
        expanded.add(node.get_state())
        children = node.get_child_nodes_no_h(state, rules)
        for c in children:
            if c.get_state() not in expanded or (c not in fringe and c.get_action != "NoOp"):
                heapq.heappush(fringe, c)

    return soln

