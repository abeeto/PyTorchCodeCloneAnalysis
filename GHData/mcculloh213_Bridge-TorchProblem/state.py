__author__ = "H.D. 'Chip' McCullough IV"

# state = (A, B, C, D, P)
# Problem state lookup table
# AKA: An hour of combinatorics, algebra, and logic
# Let X be a logical 5-tuple. Then state[X] will return an array of states that describe possible state changes given X.


state = \
{
    (False, False, False, False, False):             # "ABCDP" "" -- 00
        [
            (True, True, False, False, True),    # "CD" "ABP" -- 01
            (True, False, True, False, True),    # "BD" "ACP" -- 02
            (True, False, False, True, True),    # "BC" "ADP" -- 03
            (True, False, False, False, True),   # "BCD" "AP" -- 04
            (False, True, True, False, True),    # "AD" "BCP" -- 05
            (False, True, False, True, True),    # "AC" "BDP" -- 06
            (False, True, False, False, True),   # "ACD" "BP" -- 07
            (False, False, True, True, True),    # "AB" "CDP" -- 08
            (False, False, True, False, True),   # "ABD" "CP" -- 09
            (False, False, False, True, True)    # "ABC" "DP" -- 10
        ],
    (True, True, False, False, True):                # "CD" "ABP" -- 01
        [
            (False, True, False, False, False),  # "ACDP" "B" -- 11
            (True, False, False, False, False),  # "BCDP" "A" -- 12
            (False, False, False, False, False)  # "ABCDP" "" -- 00
        ],
    (True, False, True, False, True):                # "BD" "ACP" -- 02
        [
            (False, False, True, False, False),  # "ABDP" "C" -- 13
            (True, False, False, False, False),  # "BCDP" "A" -- 12
            (False, False, False, False, False)  # "ABCDP" "" -- 00
        ],
    (True, False, False, True, True):                # "BC" "ADP" -- 03
        [
            (False, False, False, True, False),  # "ABCP" "D" -- 14
            (True, False, False, False, False),  # "BCDP" "A" -- 12
            (False, False, False, False, False)  # "ABCDP" "" -- 00
        ],
    (True, False, False, False, True):               # "BCD" "AP" -- 04
        [
            (False, False, False, False, False)  # "ABCDP" "" -- 00
        ],
    (False, True, True, False, True):               # "AD" "BCP" -- 05
        [
            (False, False, True, False, False),  # "ABDP" "C" -- 13
            (False, True, False, False, False),  # "ACDP" "B" -- 11
            (False, False, False, False, False)  # "ABCDP" "" -- 00
        ],
    (False, True, False, True, True):                # "AC" "BDP" -- 06
        [
            (False, False, False, True, False),  # "ABCP" "D" -- 14
            (False, True, False, False, False),  # "ACDP" "B" -- 11
            (False, False, False, False, False)  # "ABCDP" "" -- 00
        ],
    (False, True, False, False, True):               # "ACD" "BP" -- 07
        [
            (False, False, False, False, False)  # "ABCDP" "" -- 00
        ],
    (False, False, True, True, True):                # "AB" "CDP" -- 08
        [
            (False, False, False, True, False),  # "ABCP" "D" -- 14
            (False, False, True, False, False),  # "ABDP" "C" -- 13
            (False, False, False, False, False)  # "ABCDP" "" -- 00
        ],
    (False, False, True, False, True):               # "ABD" "CP" -- 09
        [
            (False, False, False, False, False)  # "ABCDP" "" -- 00
        ],
    (False, False, False, True, True):               # "ABC" "DP" -- 10
        [
            (False, False, False, False, False)  # "ABCDP" "" -- 00
        ],
    (False, True, False, False, False):              # "ACDP" "B" -- 11
        [
            (True, True, True, False, True),     # "D" "ABCP" -- 15
            (True, True, False, True, True),     # "C" "ABDP" -- 16
            (True, True, False, False, True),    # "CD" "ABP" -- 01
            (False, True, True, True, True),     # "A" "BCDP" -- 17
            (False, True, True, False, True),    # "AD" "BCP" -- 05
            (False, True, False, True, True)     # "AC" "BDP" -- 06
        ],
    (True, False, False, False, False):              # "BCDP" "A" -- 12
        [
            (True, True, True, False, True),     # "D" "ABCP" -- 15
            (True, True, False, True, True),     # "C" "ABDP" -- 16
            (True, True, False, False, True),    # "CD" "ABP" -- 01
            (True, False, True, True, True),     # "B" "ACDP" -- 18
            (True, False, True, False, True),    # "BD" "ACP" -- 02
            (True, False, False, True, True)     # "BC" "ADP" -- 03
        ],
    (False, False, True, False, False):              # "ABDP" "C" -- 13
        [
            (True, True, True, False, True),     # "D" "ABCP" -- 15
            (True, False, True, True, True),     # "B" "ACDP" -- 18
            (True, False, True, False, True),    # "BD" "ACP" -- 02
            (False, True, True, True, True),     # "A" "BCDP" -- 17
            (False, True, True, False, True),    # "AD" "BCP" -- 05
            (False, False, True, True, True)     # "AB" "CDP" -- 08
        ],
    (False, False, False, True, False):              # "ABCP" "D" -- 14
        [
            (True, True, False, True, True),     # "C" "ABDP" -- 16
            (True, False, True, True, True),     # "B" "ACDP" -- 18
            (True, False, False, True, True),    # "BC" "ADP" -- 03
            (False, True, True, True, True),     # "A" "BCDP" -- 17
            (False, True, False, True, True),    # "AC" "BDP" -- 06
            (False, False, True, True, True)     # "AB" "CDP" -- 08
        ],
    (True, True, True, False, True):                 # "D" "ABCP" -- 15
        [
            (False, False, True, False, False),  # "ABDP" "C" -- 13
            (False, True, False, False, False),  # "ACDP" "B" -- 14
            (False, True, True, False, False),   # "ADP" "BC" -- 19
            (True, False, False, False, False),  # "BCDP" "A" -- 12
            (True, False, True, False, False),   # "BDP" "AC" -- 20
            (True, True, False, False, False)    # "CDP" "AB" -- 21
        ],
    (True, True, False, True, True):                 # "C" "ABDP" -- 16
        [
            (False, False, False, True, False),  # "ABCP" "D" -- 14
            (False, True, False, False, False),  # "ACDP" "B" -- 11
            (False, True, False, True, False),   # "ACP" "BD" -- 22
            (True, False, False, False, False),  # "BCDP" "A" -- 12
            (True, False, False, True, False),   # "BCP" "AD" -- 23
            (True, True, False, False, False)    # "CDP" "AB" -- 21
        ],
    (False, True, True, True, True):                 # "A" "BCDP" -- 17
        [
            (False, False, False, True, False),  # "ABCP" "D" -- 14
            (False, False, True, False, False),  # "ABDP" "C" -- 13
            (False, False, True, True, False),   # "ABP" "CD" -- 24
            (False, True, False, False, False),  # "ACDP" "B" -- 11
            (False, True, False, True, False),   # "ACP" "BD" -- 22
            (False, True, True, False, False)    # "ADP" "BC" -- 19
        ],
    (True, False, True, True, True):                 # "B" "ACDP" -- 18
        [
            (False, False, False, True, False),  # "ABCP" "D" -- 14
            (False, False, True, False, False),  # "ABDP" "C" -- 13
            (False, False, True, True, False),   # "ABP" "CD" -- 24
            (True, False, False, False, False),  # "BCDP" "A" -- 12
            (True, False, False, True, False),   # "BCP" "AD" -- 23
            (True, False, True, False, False),   # "BDP" "AC" -- 20
        ],
    (False, True, True, False, False):               # "ADP" "BC" -- 19
        [
            (True, True, True, True, True),      # "" "ABCDP" -- VICTORY
            (True, True, True, False, True),     # "D" "ABCP" -- 15
            (False, True, True, True, True)      # "A" "BCDP" -- 17
        ],
    (True, False, True, False, False):               # "BDP" "AC" -- 20
        [
            (True, True, True, True, True),      # "" "ABCDP" -- VICTORY
            (True, True, True, False, True),     # "D" "ABCP" -- 15
            (True, False, True, True, True)      # "B" "ACDP" -- 18
        ],
    (True, True, False, False, False):                # "CDP" "AB" -- 21
        [
            (True, True, True, True, True),      # "" "ABCDP" -- VICTORY
            (True, True, True, False, True),     # "D" "ABCP" -- 15
            (True, True, False, True, True)      # "C" "ABDP" -- 16
        ],
    (False, True, False, True, False):               # "ACP" "BD" -- 22
        [
            (True, True, True, True, True),      # "" "ABCDP" -- VICTORY
            (True, True, False, True, True),     # "C" "ABDP" -- 16
            (False, True, True, True, True)      # "A" "BCDP" -- 17
        ],
    (True, False, False, True, False):               # "BCP" "AD" -- 23
        [
            (True, True, True, True, True),      # "" "ABCDP" -- VICTORY
            (True, True, False, True, True),     # "C" "ABDP" -- 16
            (True, False, True, True, True)      # "B" "ACDP" -- 18
        ],
    (False, False, True, True, False):               # "ABP" "CD" -- 24
        [
            (True, True, True, True, True),      # "" "ABCDP" -- VICTORY
            (True, False, True, True, True),     # "B" "ACDP" -- 18
            (False, True, True, True, True)      # "A" "BCDP" -- 17
        ],
    (True, True, True, True, False):                # "ABCD" "P" -- NoOp
        [
            (True, True, True, True, False)      # "ABCD" "P" -- NoOp
        ],
    (False, False, False, False, True):             # "P" "ABCD" -- NoOp
        [
            (False, False, False, False, True)   # "P" "ABCD" -- NoOp
        ],
    (True, True, True, False, False):               # "DP" "ABC" -- 100
        [
            (True, True, True, True, True)       # "" "ABCDP" -- VICTORY
        ],
    (True, True, False, True, False):               # "CP" "ABD" -- 101
        [
            (True, True, True, True, True)       # "" "ABCDP" -- VICTORY
        ],
    (True, False, True, True, False):               # "BP" "ACD" -- 110
        [
            (True, True, True, True, True)       # "" "ABCDP" -- VICTORY
        ],
    (False, True, True, True, False):               # "AP" "BCD" -- 111
        [
            (True, True, True, True, True)       # "" "ABCDP" -- VICTORY
        ],
    (True, True, True, True, True):                 # "" "ABCDP" -- VICTORY
        [
            (True, True, True, True, True)       # "" "ABCDP" -- VICTORY
        ]
}
