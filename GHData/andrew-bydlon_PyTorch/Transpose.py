def Transpose(M):
    m = len(M)
    n = len(M[0])
    return [[M[j][i] for j in range(m)] for i in range(n)]


def TransposeFast(M):
    return [list(ele) for ele in zip(*M)]


def TransposeMultiprocRow(MyTuple):
    M = MyTuple[0]
    i = MyTuple[1]
    return [M[j][i] for j in range(len(M))]


def TransposeMultiproc(M, threads=24, timed=False):
    from multiprocessing import Pool

    m, n = len(M), len(M[0])

    pool = Pool(processes=threads)

    if timed:
        from time import time
        start_time = time()

    result = []

    for i in range(m//threads):
        result += pool.map(TransposeMultiprocRow, [(M, i*threads+j) for j in range(threads)])

    remainder = m % threads

    if remainder > 0:
        pool = Pool(processes=remainder)
        result += pool.map(TransposeMultiprocRow, [(m//threads)*threads+j for j in range(remainder)])

    if timed:
        from time import time
        end_time = time()
        print(end_time-start_time, "seconds.")

    return result



def LeftRotation(M):
    m = len(M)
    n = len(M[0])
    return [[M[j][n-1-i] for j in range(m)] for i in range(n)]


def RightRotation(M):
    m = len(M)
    n = len(M[0])
    return [[M[m-1-j][i] for j in range(m)] for i in range(n)]


def FlipMat(M):
    m = len(M)
    n = len(M[0])
    return [[M[m-1-i][n-1-j] for j in range(n)] for i in range(m)]


def AlternativeFlipMat(M):
    return RightRotation(RightRotation(M))


def AddInRange(M):
    myRange = input("What range would you like to increment?  Give as 'L U', \
     where L is the lower bound and U is the upper bound inclusive").split()
    lowerBound = int(myRange[0])
    upperBound = int(myRange[1])+1
    increment = float(input("By how much would you like to increment?"))

    return [M[i][:lowerBound]+list(map((lambda x: x+increment), M[i][lowerBound:upperBound]))+M[i][upperBound:] for i in range(len(M))]


def LocateFunction(L, m):
    LocalList = L.copy()
    n = LocalList.count(m)
    Occurences = []
    i=0
    while i<n:
        Occurences.append(LocalList.index(m)+i)
        LocalList.remove(m)
        i += 1
    return Occurences


def GenerateRow(n):
    import random
    return [random.random()*100 for _ in range(n)]


def RandomGeneratedMatrixMultiproc(m, n, threads=24, timed=False):
    from multiprocessing import Pool
    pool = Pool(processes=threads)

    if timed:
        from time import time
        start_time = time()

    result = []

    for i in range(m//threads):
        result += list(pool.map(GenerateRow, [n for _ in range(threads)]))

    remainder = m%threads

    if remainder > 0:
        pool = Pool(processes=remainder)
        result += list(pool.map(GenerateRow, [n for _ in range(remainder)]))

    if timed:
        from time import time
        end_time = time()
        print(end_time-start_time, "seconds.")

    return result
