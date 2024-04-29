import math


def evaluate(right, left, toMove):
    cost = max(toMove)
    right.extend(toMove)
    left = [_ for _ in left if _ not in toMove]
    returnCost = min(right)
    right.remove(returnCost)
    left.append(returnCost)
    rSum = sum(left)
    totalValue = cost + returnCost + rSum
    return totalValue, right, left, (cost + returnCost), [returnCost]


def hillClimbing(persons, movement, paint, changeText, end, right=[], left=[], total=0):
    if len(left) == 0:
        left = persons[:]
    left.sort()
    toMove = []
    minValue = math.inf
    currentCost = 0
    toReturn = []

    if len(left) <= 2:
        toMove = left
        cost = max(left)
        right.extend(left)
        nLeft = []
        nRight = right
        currentCost = cost
        minValue = currentCost

        changeText("value", minValue)
        paint(toMove, "blue")

        changeText("value", 0)
        paint(toMove, "white")

    else:
        for a in left:
            _ = left[:]
            _.remove(a)
            for b in _:
                totalValue, cRight, cLeft, totalCost, cReturn = evaluate(right[:], left[:], [a, b])

                changeText("value", totalValue)
                paint([a, b], "blue")

                if totalValue < minValue:
                    minValue = totalValue
                    toMove = [a, b]
                    nRight = cRight
                    nLeft = cLeft
                    currentCost = totalCost
                    toReturn = cReturn

                changeText("value", 0)
                paint([a, b], "white")

    print("______MOVE______")
    print(minValue)
    print(toMove)
    print(nRight)
    print(nLeft)
    print("Return: ", toReturn)

    paint(toMove, "green")

    changeText("cost", max(toMove))
    changeText("total", total + max(toMove))

    # Move to the Right
    movement(toMove, "right")

    total = total + currentCost

    # Move to the Left
    if len(toReturn) > 0:
        changeText("cost", min(toReturn))
        changeText("total", total)
        movement(toReturn, "left")
        changeText("cost", 0)

    if sum(nRight) == sum(persons):
        print("______COST______")
        print("COST: ", total)
        end(total)
    else:
        hillClimbing(persons, movement, paint, changeText, end, nRight, nLeft, total)
