import concurrent.futures

def getitem(idx):
    data=torch.load(f'Validation224x8/data{idx}.pth')
    target_availabilities=torch.load(f'Validation224x8/avail{idx}.pth')
    targets=torch.load(f'Validation224x8/targets{idx}.pth')
    
    return {
            "image": data['image'],
            "world_from_agents": data["world_from_agent"],
            "centroids" : data["centroid"],
            "targets": targets,
            "avails": target_availabilities
        }

def val_iterator(indices):
    out={'image':[],
    'world_from_agents':[],
    'centroids':[],
    'targets':[],
    'avails':[]}
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(getitem, indices)
        for result in results:
            out['image'].extend(result['image'])
            out["world_from_agents"].extend(result["world_from_agents"])
            out['centroids'].extend(result['centroids'])
            out['targets'].extend(result['targets'])
            out['avails'].extend(result['avails'])
    return out
    
# ====== # For Evaluation (Sample)


def eval_loop(model,device,total_length,batch_size,criterion):
    model.eval()
    torch.set_grad_enabled(False)
    # store information for evaluation
    losses_valid=[]
    bar = tqdm(np.arange(0,total_length-batch_size,batch_size),position=0,leave=True)
    for i in bar:
        indices=np.arange(i,i+batch_size)
        out=val_iterator(indices)
        target_availabilities = torch.stack(out['avails']).to(device)
        targets               = torch.stack(out['targets']).to(device)
        images                = torch.stack(out['image']).to(device)
