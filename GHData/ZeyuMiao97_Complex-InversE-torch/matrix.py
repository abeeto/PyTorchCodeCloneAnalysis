import random

from pandas import *

input = int(input("输入矩阵数："))

matrix = [[0] * 2 for i in range(input)]

for i in range(input):  # 生成矩阵

    if i == 0:

        matrix[i][0] = random.randrange(100)

        matrix[i][1] = random.randrange(100)

    else:

        matrix[i][0] = matrix[i - 1][1]

        matrix[i][1] = random.randrange(100)

m = [[0] * input for i in range(input)]  # 记录连乘次数

s = [[0] * input for j in range(input)]  # 记录括号位置


def MatrixMultiplication(inp):
    for i in range(inp):
        m[i][i] = 0

    for r in range(1, inp):

        for i in range(inp - r):

            j = i + r

            m[i][j] = m[i + 1][j] + matrix[i][0] * matrix[i][1] * matrix[j][1]

            s[i][j] = i + 1

            for k in range(i + 1, j):

                judge = m[i][k] + m[k + 1][j] + matrix[i][0] * matrix[k][1] * matrix[j][1]

                if judge < m[i][j]:
                    m[i][j] = judge

                    s[i][j] = k + 1


def printmatrix(left, right):
    if left == right:

        print("A" + str(left + 1), end='')

    else:

        print("(", end='')

        printmatrix(left, s[left][right] - 1)

        printmatrix(s[left][right], right)

        print(")", end='')


MatrixMultiplication(input)

dm = DataFrame(m, index=list(range(1, input + 1)), columns=list(range(1, input + 1)))

ds = DataFrame(s, index=list(range(1, input + 1)), columns=list(range(1, input + 1)))

print(matrix)

print("数乘次数：\n", dm)

print("括号位置：\n", ds)

print("最终结果：")

printmatrix(0, input - 1)