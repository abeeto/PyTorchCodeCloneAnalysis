__author__ = "H.D. 'Chip' McCullough IV"

# action - {hash: [action, path_cost, heuristic_cost]}
action = \
    {
        0: ["CROSS: A, B", 2, 1],    # B dominates A
        1: ["CROSS: A, C", 5, 1],    # C dominates A
        2: ["CROSS: A, D", 10, 1],   # D dominates A
        3: ["CROSS: B, C", 5, 2],    # C dominates B
        4: ["CROSS: B, D", 10, 2],   # D dominates B
        5: ["CROSS: C, D", 10, 5],   # D dominates C
        6: ["CROSS: A", 1, 1],       # A alone
        7: ["CROSS: B", 2, 2],       # B alone
        8: ["CROSS: C", 5, 5],       # C alone
        9: ["CROSS: D", 10, 10],     # D alone
        10: ["NoOp", 0, 0]           # NoOp
    }

# Let X be a logical 5-tuple. Then rules[X] will return an array of actions that can be taken in the given state X.
rules = \
    {
        (False, False, False, False, False):  # "ABCDP" "" -- 00
            [
                action[0],  # AB
                action[1],  # AC
                action[2],  # AD
                action[6],  # A
                action[3],  # BC
                action[4],  # BD
                action[7],  # B
                action[5],  # CD
                action[8],  # C
                action[9]   # D
            ],
        (True, True, False, False, True):     # "CD" "ABP" -- 01
            [
                action[6],  # A
                action[7],  # B
                action[0]   # AB
            ],
        (True, False, True, False, True):     # "BD" "ACP" -- 02
            [
                action[6],  # A
                action[8],  # C
                action[1]   # AC
            ],
        (True, False, False, True, True):     # "BC" "ADP" -- 03
            [
                action[6],  # A
                action[9],  # D
                action[2]   # AD
            ],
        (True, False, False, False, True):    # "BCD" "AP" -- 04
            [
                action[6]  # A
            ],
        (False, True, True, False, True):     # "AD" "BCP" -- 05
            [
                action[7],  # B
                action[8],  # C
                action[3]   # BC
            ],
        (False, True, False, True, True):     # "AC" "BDP" -- 06
            [
                action[7],  # B
                action[9],  # D
                action[4]   # BD
            ],
        (False, True, False, False, True):    # "ACD" "BP" -- 07
            [
                action[7]  # B
            ],
        (False, False, True, True, True):     # "AB" "CDP" -- 08
            [
                action[8],  # B
                action[9],  # D
                action[4]   # BD
            ],
        (False, False, True, False, True):    # "ABD" "CP" -- 09
            [
                action[8]  # C
            ],
        (False, False, False, True, True):    # "ABC" "DP" -- 10
            [
                action[9]  # D
            ],
        (False, True, False, False, False):   # "ACDP" "B" -- 11
            [
                action[1],  # AC
                action[2],  # AD
                action[6],  # A
                action[5],  # CD
                action[8],  # C
                action[9]   # D
            ],
        (True, False, False, False, False):   # "BCDP" "A" -- 12
            [
                action[3],  # BC
                action[4],  # BD
                action[7],  # B
                action[5],  # CD
                action[8],  # C
                action[9]   # D
            ],
        (False, False, True, False, False):   # "ABDP" "C" -- 13
            [
                action[0],  # AB
                action[2],  # AD
                action[6],  # A
                action[4],  # BD
                action[7],  # B
                action[9]   # D
            ],
        (False, False, False, True, False):   # "ABCP" "D" -- 14
            [
                action[0],  # AB
                action[1],  # AC
                action[6],  # A
                action[3],  # BC
                action[7],  # B
                action[8]   # C
            ],
        (True, True, True, False, True):      # "D" "ABCP" -- 15
            [
                action[0],  # AB
                action[1],  # AC
                action[6],  # A
                action[3],  # BC
                action[7],  # B
                action[8]   # C
            ],
        (True, True, False, True, True):      # "C" "ABDP" -- 16
            [
                action[0],  # AB
                action[2],  # AD
                action[6],  # A
                action[4],  # BD
                action[7],  # B
                action[9]   # D
            ],
        (False, True, True, True, True):      # "A" "BCDP" -- 17
            [
                action[3],  # BC
                action[4],  # BD
                action[7],  # B
                action[5],  # CD
                action[8],  # C
                action[9]   # D
            ],
        (True, False, True, True, True):      # "B" "ACDP" -- 18
            [
                action[1],  # AC
                action[2],  # AD
                action[6],  # A
                action[5],  # CD
                action[8],  # C
                action[9]   # D
            ],
        (False, True, True, False, False):    # "ADP" "BC" -- 19
            [
                action[2],  # AD
                action[6],  # A
                action[9]   # D
            ],
        (True, False, True, False, False):    # "BDP" "AC" -- 20
            [
                action[4],  # BD
                action[7],  # B
                action[9]   # D
            ],
        (True, True, False, False, False):    # "CDP" "AB" -- 21
            [
                action[5],  # CD
                action[8],  # C
                action[9]   # D
            ],
        (False, True, False, True, False):    # "ACP" "BD" -- 22
            [
                action[1],  # AC
                action[6],  # A
                action[8]   # C
            ],
        (True, False, False, True, False):    # "BCP" "AD" -- 23
            [
                action[3],  # BC
                action[7],  # B
                action[8]   # C
            ],
        (False, False, True, True, False):    # "ABP" "CD" -- 24
            [
                action[0],  # AB
                action[6],  # A
                action[7]   # B
            ],
        (True, True, True, True, False):      # "P" "ABCD" -- NoOp
            [
                action[10]  # NoOp
            ],
        (False, False, False, False, True):   # "ABCD" "P" -- NoOp
            [
                action[10]  # NoOp
            ],
        (True, True, True, False, False):     # "DP" "ABC" -- 100
            [
                action[9]  # D
            ],
        (True, True, False, True, False):     # "CP" "ABD" -- 101
            [
                action[8]  # C
            ],
        (True, False, True, True, False):     # "BP" "ACD" -- 110
            [
                action[7]  # B
            ],
        (False, True, True, True, False):     # "AP" "BCD" -- 111
            [
                action[6]  # A
            ],
        (True, True, True, True, True):       # "" "ABCDP" -- VICTORY
            [
                action[10]  # NoOp
            ]
    }
