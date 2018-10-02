import numpy as np

dataset = np.genfromtxt('world_alcohol.txt',delimiter=',', dtype=str, skip_header=1)

print(dataset[1,4])

li = np.array([[1,2,3],
              [2,6,5]])
print(li.shape)


print(np.arange(16).reshape(4,4))

print(np.ones((3,4), dtype=np.int))

print(np.arange(10,50,5))

print(np.random.random((2,3)))

from  numpy import pi

print(pi)
print(np.linspace(0, 2 * pi,100))

a = np.array([20,30,40,50])
b = np.arange(4)
print(a-b)
print(a-1)
print(a**2)

A = np.array([[1,2],
              [2,3]])
B = np.array([[-3,2],
             [2,-1]])
print(A*B)
print(A.dot(B))
#矩阵乘法
print(np.dot(A,B))
print(np.exp(B))

a = np.floor(10 * np.random.random((3,4)))
print(a)
#化为一维数组
print(a.ravel())
#指定列行
print(a.reshape(2,-1))
#转置
print(a.T)

a = np.floor(10 * np.random.random((2,2)))
b = np.floor(10 * np.random.random((2,2)))

print(a)
print(b)
print(np.hstack((a,b)))