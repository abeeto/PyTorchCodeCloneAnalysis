from turtle import color
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(74)
X = np.random.rand(20,1) * 10
y = np.random.randint(0, 10, size=20)
plt.scatter(X, y, color='red')

class KNN:
    def __init__(self, X, y, k) -> None:
        self.X = X
        self.y = y
        self.k = k
    
    def class_point(self, point):
        distance = []
        for p in self.X:
            dis = np.mean(np.square(p - point))
            distance.append(dis.item())
        sorted_dis = sorted(enumerate(distance), key=lambda x: x[1], reverse=True)
        labels_list = self.y[[item[0] for item in sorted_dis[: self.k]]]
        label = sorted(dict(zip(*np.unique(labels_list, return_counts=True))).items(), key=lambda x: x[1], reverse=True)[0][0]
        return label



if __name__ == '__main__':
    knn = KNN(X, y, 5)
    point = np.random.rand(1, 1) * 10
    label = knn.class_point(point)
    print('point: ', point.item(), ', label: ', label)
    plt.scatter(point, label, color='blue')
    plt.show()
