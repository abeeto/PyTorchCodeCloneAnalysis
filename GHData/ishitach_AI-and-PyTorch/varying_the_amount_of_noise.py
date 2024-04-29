def query(db):
    true_result=torch.mean(db.float())
    first_coin_flip = (torch.rand(len(db)) > 0.5).float()
    second_coin_flip = (torch.rand(len(db)) > 0.5).float()
    augmented_db=db.float()*first_coin_flip+(1-first_coin_flip)*second_coin_flip
    db_result = torch.mean(augmented_db.float())*2-0.5
    return db_result,true_result
    

query(db)

db,pdbs=create_db_and_parallel(10)
private_result, result = query(db)
print("With Noise:"+ str(private_result))
print("Without Noise:"+ str(result))


db,pdbs=create_db_and_parallel(100)
private_result, result = query(db)
print("With Noise:"+ str(private_result))
print("Without Noise:"+ str(result))

db,pdbs=create_db_and_parallel(1000)
private_result, result = query(db)
print("With Noise:"+ str(private_result))
print("Without Noise:"+ str(result))

db,pdbs=create_db_and_parallel(10000)
private_result, result = query(db)
print("With Noise:"+ str(private_result))
print("Without Noise:"+ str(result))

db,pdbs=create_db_and_parallel(100000)
private_result, result = query(db)
print("With Noise:"+ str(private_result))
print("Without Noise:"+ str(result))

#function with noise
def query(db, noise=0.2):
    noise=0.2
    true_result=torch.mean(db.float())
    first_coin_flip = (torch.rand(len(db)) > noise).float()
    second_coin_flip = (torch.rand(len(db)) > 0.5).float()
    augmented_db=db.float()*first_coin_flip+(1-first_coin_flip)*second_coin_flip
    db_result = torch.mean(augmented_db.float())*2-0.5

    skewed_res = augmented_db.float().mean()
    private_result=((skewed_res)/noise - 0.5) * noise/(1-noise)
    return db_result,true_result


noise = 0.5
true_dis_mean=0.7
noise_dis_mean=0.5

augmented_dis_mean = (true_dis_mean* noise + noise_dis_mean*(1-noise))
augmented_dis_mean


db,pdbs=create_db_and_parallel(100)
private_result, result = query(db,noise=0.1)
print("With Noise:"+ str(private_result))
print("Without Noise:"+ str(result))

db,pdbs=create_db_and_parallel(100)
private_result, result = query(db,noise=0.2)
print("With Noise:"+ str(private_result))
print("Without Noise:"+ str(result))


db,pdbs=create_db_and_parallel(100)
private_result, result = query(db,noise=0.4)
print("With Noise:"+ str(private_result))
print("Without Noise:"+ str(result))
