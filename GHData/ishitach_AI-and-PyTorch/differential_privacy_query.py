def sum_query(db):
    return db.sum()
    
sum(db)

def laplacian_mechanism(db, query, sensitivity):
    beta= sensitivity/epsilon
    noise= torch.tensor(np.random.laplace(0,beta,1))
    return query(db) + noise

laplacian_mechanism(db, sum_query, 1)

def mean_query(db):
    return torch.mean(db.float())
    
laplacian_mechanism(db, mean_query, 1/100)    
