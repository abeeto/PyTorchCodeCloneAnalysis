#まずはsoftmax関数から

#ラベル＝４つ

#e = 2.71828182846

#10,5,80,2

dataList = []

dataList.append(2)
dataList.append(3)
dataList.append(1.5)
dataList.append(1)

#print(dataList)
#print(len(dataList))
#print(dataList[0])

def softmax(dataList):
    e = 2.71828182846
    i = len(dataList)
    a = []
    b = 0.0
    cnt = 0
    #分子
    while True:
        if cnt == i:
            break
        data = int(dataList[cnt])
        j = 0
        out = 1.0
        while True:
            if j == data:
                break
            out *= e
            j += 1
        a.append(out)
        cnt += 1
    #分母
    cnt = 0
    while True:
        if cnt == i:
            break
        b += a[cnt]
        cnt += 1
    #計算
    f = 0.0
    F = []
    cnt = 0
    while True:
        if cnt == i:
            break
        f = a[cnt] / b
        F.append(f)
        #print(f)
        cnt += 1
    return F

def Euclidean(m,n):
    if m < n:
        m,n = n,m
    if n == 0:
        return m
    elif m % n == 0:
        return n
    else:return Euclidean(n,m%n)

def deciToFrac(deci):
    i = 0
    while deci % 1 != 0:
        deci = deci * 10
        i += 1

    numer,denomi = int(deci),int(10 ** i)
    
    gcd = Euclidean(numer,denomi)

    return numer / gcd,denomi / gcd

def log(dataList):
    # logb C = P  bln(P) = C
    # 10ln(P) = Mになればよい
    #print(deciToFrac(0.03))
    #print(deciToFrac(0.03)[0])
    l = len(dataList)
    #print(l)
    cnt = 0
    minusX = [] #0-1
    minusX.append(1)
    minusX.append(1)
    minusX.append(1)
    minusX.append(1)
    while True:
        if cnt == l:
            break
        print(dataList[cnt])
        print((10**(-deciToFrac(minusX[cnt])[0]/deciToFrac(minusX[cnt])[1])))
        if (dataList[cnt]) < (10**(-deciToFrac(minusX[cnt])[0]/deciToFrac(minusX[cnt])[1])):
            print("もっと小さく")
            pass 
        elif (dataList[cnt]) > (10**(-deciToFrac(minusX[cnt])[0]/deciToFrac(minusX[cnt])[1])):
            print("もっと大きく")
            pass 
        else:
            pass 
        cnt += 1
        
    pass

#softmax(dataList)
log(softmax(dataList))
#print(10**(-30/10)) == M






