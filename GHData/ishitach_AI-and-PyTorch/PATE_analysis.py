labels = np.array([9,9,3,6,9,9,9,9,8,2])
counts = np.bincount(labels, minlength =10)
counts

query_result = np.argmax(counts)
query_result

!pip install pysft

from syft.frameworks.torch.differential_privacy import pate

import numpy as np
num_teachers, num_examples, num_labels = (100,100,10)
preds = (np.random.rand(num_teachers, num_examples)*num_labels).astype(int)
indices = (np.random.rand(num_examples)*num_labels).astype(int)

date_dep_eps, date_ind_eps = pate.perform_analysis(teacher_preds=preds,indices= indices,noise_eps=0.1,delta=1e-5 )

date_dep_eps
#11.756462732485105

date_ind_eps
#11.756462732485115

preds[:,0:5] *= 0

date_dep_eps, date_ind_eps = pate.perform_analysis(teacher_preds=preds,indices= indices,noise_eps=0.1,delta=1e-5 )

date_dep_eps
#7.867987172744542

date_ind_eps
#11.756462732485115

preds[:,0:50] *= 0

date_dep_eps, date_ind_eps = pate.perform_analysis(teacher_preds=preds,indices= indices,noise_eps=0.1,delta=1e-5 )
#Warning: May not have used enough values of l. Increase 'moments' variable and run again.

date_dep_eps
#1.52655213289881

date_ind_eps
#11.756462732485115
