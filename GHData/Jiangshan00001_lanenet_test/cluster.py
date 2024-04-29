from sklearn.cluster import DBSCAN
import numpy as np
test_data=[ [ [1.0,0.0,0.0  ], [ 0.9,0,0], [0.8,0,0] ],
            [[1.0,0.0,-3.5  ], [ 0.9,0,-3.5], [0.8,0,-3.5]],
            [[1.0,0.0,-7.5  ], [ 0.9,0,-7.5], [0.8,0,-7.5]],
            [[1.0,0.0,-17.5  ], [ 0.9,0,-17.5], [0.8,0,-17.5]],
            ]

test_data=[ [[1,2,3 ], [4,5,6]], [ [7,8,9], [10,11,12]] ]
test_data=np.array(test_data)

a=test_data.view(-1)
print(a)

hotel_cluster = DBSCAN(eps=3, min_samples=2, )
db = hotel_cluster.fit(test_data)
db.labels_
