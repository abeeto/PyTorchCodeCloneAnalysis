import numpy as np

num_teachers=10
num_examples=10000
num_labels = 10

preds = (np.random.rand(num_teachers, num_examples)*num_labels).astype(int)

preds[0].shape

preds[:,0]

#fake - for experimentation
preds = (np.random.rand(num_teachers, num_examples)*num_labels).astype(int)

new_labels = list()

for an_image in preds:
    label_counts = np.bincount(an_image, minlength = num_labels)
    epsilon=0.1
    beta = 1/epsilon
    
    for i in range(len(label_counts)):
        label_counts[i] +=np.random.laplace(0,beta,1)
        
    new_label = np.argmax(label_counts)
    new_labels.append(new_label)
    
len(new_labels)    
