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
db,pdbs = create_db_and_parallel(5000)

def query(db):
    return db.sum()
    
def sensitivity(query, n_entries=1000):
    db,pdbs = create_db_and_parallel(n_entries)
    full_db_result = query(db)
    max_dis=0
    for i in pdbs:
        pdb_result = query(i)
        db_dist= torch.abs(pdb_result-full_db_result)

        if(db_dist> max_dis):
            max_dis = db_dist

    return max_dis   

sensitivity(query)

def query(db):
    return db.float().mean()
    
sensitivity(query)


def query(db, thershold=5):
    return (db.sum() > thershold).float()


db,pdbs = create_db_and_parallel(10)
db.sum()

query(db)

for i in range(10):
    sens_q = sensitivity(query, n_entries=10)
    print(sens_q)
    

db,_ = create_db_and_parallel(100)

pdb=get_parallel_db(db, remove_index=10)

sum(db)-sum(pdb)

(sum(db).float()/len(db))-(sum(pdb).float()/len(pdb))

(sum(db).float()>49)-(sum(pdb).float()>49)

