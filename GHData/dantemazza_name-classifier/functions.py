import const
import random as rd


def popall(name):
    const.name_map.pop(name)
    const.M_names.remove(name)
    const.F_names.remove(name)

#extract a subset of the dataset with an option for labelled
def extract(size, labelled=False):
    half = int(size/2)

    result = {} if labelled else []
    for i in range(half):
        m, f = len(const.M_names), len(const.F_names)
        index = rd.randint(0, f-1 if m > f else m-1)
        if labelled:
            result[const.M_names[index]] = 1
            result[const.F_names[index]] = 0
        else:
            result.append(const.M_names[index])
            result.append(const.F_names[index])
        del const.M_names[index]
        del const.F_names[index]
    return result

