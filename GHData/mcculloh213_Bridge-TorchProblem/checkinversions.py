__author__ = "H.D. 'Chip' McCullough IV"


def check_inversions(state):
    top = state[0]
    mid = state[1]
    low = state[2]
    inv = []
    solvable = False
    inversions = 0
    for i in top:
        if int(i) != 0:
            inv.append(int(i))
    for j in mid:
        if int(j) != 0:
            inv.append(int(j))
    for k in low:
        if int(k) != 0:
            inv.append(int(k))
    for l in range(0, len(inv)):
        for m in range(l+1, len(inv)):
            if inv[m] > inv[l]:
                inversions += 1
    if inversions % 2 == 0:
        solvable = True

    return True
