import numpy as np

a = np.array([1,2,3])

print (a.shape)

b = np.array([[1,2,3],[4,5,6],[4,5,6],[4,5,6]])

print('%s %s' %(" fila columna ", b.shape))

C = np.zeros((3,2))

D = np.ones((10,3))

print(C)

print(D)

E = np.full((2,3),7)

print( E)

I = np.eye(2)

print(I)

R = np.random.randint(0,10,(3,4))

print(R)

print(R[:,:3])

print(R[0,0])

A = np.array([[1,2],[3,4],[5,6],[7,8]])

print (A)

B =np.array([1,0,1,0])

A[np.arange(4),B]+=10
print(A )

print(A[np.arange(4),B]+10)


print( A[[0,1,2],[0,1,0]] )



bool2 = (A>2)

print(bool2)

print(A[bool2])

print(A[A>6])

print(A[0,0],A[1,1],A[2,0])


A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])

print(A+B)
print(np.add(A,B))
print(A-B)
print(np.subtract(A,B))
print(A*B)
print(np.multiply(A,B))
print(A/B)
print(np.divide(A,B))
print(np.sqrt(A))

print(A)
print(A.dot(np.array([[1],[2]])))

print(np.sum(A.dot(np.array([[1],[2]]))))

B = np.array([[5,6],[7,8],[9,0]])
print(A.dot(B.T))

x = np.array([[1,2,3],[1,2,3],[1,2,3]])
v  = np.array([1,2,3])
y= np.empty_like(x)

for i in range(3):
    y[i,:] = x[i,:]+v

print(x)
print(y)