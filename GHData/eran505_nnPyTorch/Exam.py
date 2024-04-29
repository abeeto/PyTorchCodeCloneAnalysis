# Efrat Hershkovich
# 208286716

import random
import numpy as np
from scipy.stats import linregress

###########  1  ###############

def f1(l1 ,l2 ,l3):
    ans =[]
    for x1 in l1:
        if x1 not in l2 and x1 not in l3:
            if x1 not in ans:
                ans.append(x1)
    for x2 in l2:
        if x2 not in l1 and x2 not in l3:
            if x2 not in ans:
                ans.append(x2)
    for x3 in l3:
        if x3 not in l1 and x3 not in l2:
            if x3 not in ans:
                ans.append(x3)
    return ans

###########  2 ###############

class C1:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def abs(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __mul__(self, other):
        x1 = self.x * other.x
        y1 = self.x * other.y
        y2 = self.y * other.x
        x2 = (self.y * other.y) * -1
        if y1 + y2 == 0:
            return C2((x1 + x2), (y1 + y2))
        else:
            return C1((x1 + x2), (y1 + y2))


class C2(C1):
    def __init__(self, x, y):
        super().__init__(x, y)

    def __mul__(self, other):
        x1 = self.x * other.x
        y1 = self.x * other.y
        y2 = self.y * other.x
        x2 = (self.y * other.y) * -1
        if y1 + y2 == 0:
            return C2((x1 + x2), (y1 + y2))
        else:
            return C1((x1 + x2), (y1 + y2))

###########  3 ###############
def f2(number):
    if number == 2:
        return True
    for i in range(2 ,number -1):
        if i>= number:
            return False
        if number % i == 0:
            return False
    return True

###########  4 ###############
def f3(dico):
    ans=[]
    for key,value in dico.items():
        for small_list in value:
            for x in small_list:
                ans.append(x)
    return ans

###########  5 ###############

def f4(n):
    random.seed(1)
    np.random.seed(1)
    maxi=100000
    sums =  0
    lst = [ x for x in range(52)]
    rnd = np.random.choice(lst,3)
    for k in range(maxi):
        ctr = 0
        for i in range(n):
            f = random.choice(lst)
            if f in rnd:
                ctr = ctr +1
                break
        if ctr:
            sums = sums +1
    return sums/maxi


############## 6 ####################

class Student(object):

    def __init__(self, name, features):
        self.name = name
        self.features = features

    def __str__(self):
        return self.name +':'+ str(self.features)

def euclidean_distance(vec_1,vec_2):
    sum=0
    for i in range(len(vec_1)):
        sum+=(vec_1[i]-vec_2[i])**2
    return sum**0.5

def f5(l_sutdents):
    random.seed(1)
    np.random.seed(1)
    size = len(l_sutdents)
    idx_list = np.random.choice(size, 3)
    L = [ l_sutdents[idx_list[0]], l_sutdents[idx_list[1]], l_sutdents[idx_list[2]] ]
    l1=[ l_sutdents[idx_list[0]] ]
    l2=[ l_sutdents[idx_list[1]] ]
    l3=[ l_sutdents[idx_list[2]] ]
    for student_i in l_sutdents:
        d1 = euclidean_distance(student_i.features,L[0].features)
        d2 = euclidean_distance(student_i.features, L[1].features)
        d3 = euclidean_distance(student_i.features, L[2].features)
        min_d = min(d1,d2,d3)
        if d1 == min_d:
            l1.append(student_i)
        elif d2 == min_d:
            l2.append(student_i)
        else:
            l3.append(student_i)
    return l1,l2,l3


###########  6 ###############

def dissmilarty(list_student):
    sum=0
    for i in range(len(list_student)):
        for j in range(i+1,len(list_student)):
            d = euclidean_distance(list_student[i],list_student[j])
            sum+=d
    return sum/len(list_student)

def f6(l1,l2,l3):
    x1 = dissmilarty(l1)
    x2 = dissmilarty(l2)
    x3 = dissmilarty(l3)
    return x1+x2+x3

###########  7 ###############


def f7(l):
    if 0==len(l):
        return ""
    return str(l[0])+f7(l[1:])

###########  8 ###############
comp1="O(n)"

###########  9 ###############
comp2="O(n**4)"

############## 10 ############
def f8(l1,l2):
    slope, intercept, r, p, se = linregress(l1, l2)
    return slope

############## 11 ############

def f9(n,a_0,a_1,a_2):
    list_item=[a_0,a_1,a_2]
    ctr=3
    while ctr <= n:
        list_item.append( list_item[ctr-1]*2+3*list_item[ctr-3])
        ctr=ctr+1
    return list_item[-1]



