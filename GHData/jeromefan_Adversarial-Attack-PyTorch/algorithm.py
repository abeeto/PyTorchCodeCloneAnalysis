from fgsm import FGSM
from pgd import PGD
from mifgsm import MIFGSM


def setupAlgorithm(chosenAlgorithm):

    if chosenAlgorithm == 'FGSM':
        return FGSM

    elif chosenAlgorithm == 'PGD':
        return PGD

    elif chosenAlgorithm == 'MI-FGSM':
        return MIFGSM

    else:
        raise Exception("其他攻击算法暂不支持! 请使用README中所述的已支持攻击算法！")
