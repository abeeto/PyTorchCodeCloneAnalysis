#The codes in differential attack file have to be runned to get the reference of fucntions

db,pdbs=create_db_and_parallel(100)

def query(db):
    true_result=torch.mean(db.float())
    first_coin_flip = (torch.rand(len(db)) > 0.5).float()
    second_coin_flip = (torch.rand(len(db)) > 0.5).float()
    augmented_db=db.float()*first_coin_flip+(1-first_coin_flip)*second_coin_flip
    db_result = torch.mean(augmented_db.float())*2-0.5
    return db_result
    
query(db)    
