#在这里练习使用dbscan
from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np
x1=np.random.normal(1,2,(30,5))
x2=np.random.normal(10,2,(30,5))
X=np.zeros((60,5))
X[:30]=x1
X[30:]=x2
#print(X)
db=DBSCAN(eps=4,min_samples=3).fit(X)
labels=db.labels_
print(labels)
score=metrics.silhouette_score(X,labels)
print(score)