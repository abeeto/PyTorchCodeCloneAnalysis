import sys
from checkinversions import check_inversions
from AStarAgent import astar, greedysearch
from UCAgent import ucs

__author__ = "H.D. 'Chip' McCullough IV"


def test_harness(initial, agent):
    goal = (True, True, True, True, True)  # Define goal state
    solution = False
    if "ucsearch" == agent.lower() or "astarsearch" == agent.lower():
        if "ucsearch" == agent.lower():
                solution = ucs(initial, goal)
        elif "astarsearch" == agent.lower():
            solution = astar(initial, goal)
        else:
            print("Well this is embarassing.")
            print("ERROR: UNCONTROLLED SEQUENCE, SYSTEM EXIT.")
            sys.exit(1)
        statelist = []
        actionlist = []
        gc = []
        hc = []
        fc = []
        if solution:  # Solution is of type Node(state, parent, action, g, depth)
            print("SOLUTION FOUND AT DEPTH {0} IN A TIME OF {1} MINUTE(S)".format(solution.get_depth(),
                                                                                  solution.get_path_cost()))
            print("---------------------------------------------------------------------------------------------------")
            while solution.get_parent() is not None:
                statelist.append(solution.get_state())  # Get states from goal
                actionlist.append(solution.get_action())  # Get actions from goal
                gc.append(solution.get_path_cost())  # Get path cost
                hc.append(solution.get_heuristic_cost())  # Get heuristic cost
                fc.append(solution.get_f_cost())  # Get f_cost
                solution = solution.get_parent()  # Traverse back up the tree
            statelist.append(solution.get_state())  # Get start state
            actionlist.append(solution.get_action())  # Get start action
            gc.append(solution.get_path_cost())  # Get start path cost
            hc.append(solution.get_heuristic_cost())  # Get start heuristic cost
            fc.append(solution.get_f_cost())  # Get start f_cost
            statelist.reverse()
            actionlist.reverse()
            gc.reverse()
            hc.reverse()
            fc.reverse()
            for i in range(0, len(statelist)):
                print("|{0}: {1} --- {2} ~ [g(n) = {3}, h(n) = {4}, f(n) = {5}])".format(i + 1, statelist[i],
                                                                                         actionlist[i], gc[i], hc[i],
                                                                                         fc[i]))
            print("---------------------------------------------------------------------------------------------------")
        else:
            print("NO SOLUTION FOUND")
    else:
        if check_inversions(initial):
            t = initial[0]
            m = initial[1]
            l = initial[2]
            te = "1 2 3".split()
            me = "8 0 4".split()
            le = "7 6 5".split()
            statelist = []
            actionlist = []
            hc = []
            solution = greedysearch([t, m, l], [te, me, le])
            print("SOLUTION FOUND AT DEPTH {0}.".format(solution.get_depth()))
            print("---------------------------------------------------------------------------------------------------")
            while solution.get_parent() is not None:
                statelist.append(solution.get_class())
                actionlist.append(solution.get_action())
                hc.append(solution.get_heuristic_cost())
                solution = solution.get_parent()
            statelist.append(solution.get_class())
            actionlist.append(solution.get_action())
            hc.append(solution.get_heuristic_cost())
            statelist.reverse()
            actionlist.reverse()
            hc.reverse()
            for i in range(0, len(statelist)):
                print("\t{0}: ACTION: {1}, HEURISTIC: {2}".format(i+1, actionlist[i], hc[i]))
                statelist[i].print_state()
                print("\n---------------------------------------------------------------------------------------------")
        else:
            print("THE INITIAL STATE IS UNSOLVABLE.")
