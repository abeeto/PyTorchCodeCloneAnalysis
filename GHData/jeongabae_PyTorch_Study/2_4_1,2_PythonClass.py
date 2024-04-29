#https://wikidocs.net/28
#함수(function)과 클래스(Class)의 차이
#1. 함수(function)로 덧셈기 구현하기
result = 0
def add(num):
    global result
    result += num
    return result

print(add(3)) #3
print(add(4)) #7

#2. 함수(function)로 두 개의 덧셈기 구현하기
result1 = 0
result2 = 0

def add1(num):
    global result1
    result1 += num
    return result1

def add2(num):
    global result2
    result2 += num
    return result2

print(add1(3))#3
print(add1(4))#7
print(add2(3))#3
print(add2(7))#10

#서로의 값에 영향을 주지않고 서로 다른 연산을 하고 있음을 볼 수 있습니다. 그렇다면 이런 두 개의 덧셈기를 클래스로 만들면 어떻게 될까요?

