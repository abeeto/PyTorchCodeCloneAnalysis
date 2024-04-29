import torch
db = torch.rand(5000)>0.5

def get_parallel_db(db, remove_index):
    return torch.cat((db[0:remove_index],db[remove_index+1:]))
get_parallel_db(db,5000)

def get_parallel_dbs(db):
    parallel_dbs=list()
    
    for i in range(len(db)):
        pdb=get_parallel_db(db,i)
        parallel_dbs.append(pdb)
    return parallel_dbs        
    
pdbs = get_parallel_dbs(db)

def create_db_and_parallel(num_entries):
    db = torch.rand(num_entries)>0.5
    pdbs = get_parallel_dbs(db)
    return db,pdbs
db,pdbs = create_db_and_parallel(20)

db

pdbs
