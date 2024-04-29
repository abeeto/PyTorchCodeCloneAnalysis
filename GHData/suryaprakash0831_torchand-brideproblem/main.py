count = 0


def two_left_case(a_side, b_side):
    move_from_a_to_b(a_side[0], a_side[1], a_side, b_side)


def three_left_case(a_side, b_side):
    move_from_a_to_b(max(a_side), min(a_side), a_side, b_side)
    move_from_b_to_a(min(b_side), a_side, b_side)
    print("Move %s back to A side" % str(min(b_side)))


def four_or_more_case(a_side, b_side):
    move_from_a_to_b(min(a_side), sorted(a_side)[1], a_side, b_side)
    move_from_b_to_a(min(b_side), a_side, b_side)
    move_from_a_to_b(max(a_side), sorted(a_side)[-2], a_side, b_side)
    move_from_b_to_a(min(b_side), a_side, b_side)


def move_from_a_to_b(first_number, second_number, a_side, b_side):
    global count
    a_side.remove(first_number)
    a_side.remove(second_number)
    b_side.append(first_number)
    b_side.append(second_number)
    count += max(first_number, second_number)
    print("Move %s & %s to B side." % (str(first_number), str(second_number)))


def move_from_b_to_a(number, a_side, b_side):
    global count
    b_side.remove(number)
    a_side.append(number)
    count += number
    print("Move %s back to A side." % str(number))


if __name__ == '__main__':
    a_side = [int(x) for x in input("Enter time of each person: ").split()]
    b_side = []
    while len(a_side) != 0:
        if len(a_side) == 1:
            print("Just bring that number to B...")
        if len(a_side) == 2:
            two_left_case(a_side, b_side)
            print("Total step: %s" % str(count))
            break
        elif len(a_side) == 3:
            three_left_case(a_side, b_side)
        else:
            four_or_more_case(a_side, b_side)
