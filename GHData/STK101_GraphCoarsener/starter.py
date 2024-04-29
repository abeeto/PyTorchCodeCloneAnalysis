import math
import torch
import matplotlib.pyplot as plt
from torch_geometric.datasets import Flickr
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Flickr
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import CitationFull
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import GitHub
from torch_geometric.datasets import Reddit2
from torch_geometric.datasets import Planetoid
import argparse
import numpy as np
from trainCoarse import run
import graph_attacker
import logging
import sys
import random
from deeprobust.graph.data import Dpr2Pyg, Pyg2Dpr
from deeprobust.graph.utils import sparse_mx_to_torch_sparse_tensor, preprocess
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from deeprobust.graph.data import Dataset, PrePtbDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Coarsened Graph Training')
    parser.add_argument('--data_choice',type=int,required=True,help="Path to the dataset")
    parser.add_argument('--bw',type=float,required=True,help="bin width that is reduction ratio")
    parser.add_argument('--epochs' , type=int,required=False, default=300,help="Number of epochs to train the original graph")
    parser.add_argument('--normalize' , type = bool, required=False, default=False,help="Normalize the features")
    parser.add_argument('--reps', type = int,default= 5, help = "Number of repeations before reporting final accuracy")
    parser.add_argument('--gsp', type = bool,default= False, help = "Using GSP methods to refine coarsened graph adjacency matrix")
    parser.add_argument('--plot', type = bool, default= False, help = "PLotting a b/w vs coarsening acc and b/w vs coarsening ratio " )
    parser.add_argument('--lr', type = float, default= -1, help = "PLotting a b/w vs coarsening acc and b/w vs coarsening ratio " )
    parser.add_argument('--decay', type = float, default= -1, help = "PLotting a b/w vs coarsening acc and b/w vs coarsening ratio " )
    parser.add_argument('--tuning_gcn', type = bool, default= False, help = "PLotting a b/w vs coarsening acc and b/w vs coarsening ratio " )
    parser.add_argument('--tuning_opt', type = bool, default= False, help = "PLotting a b/w vs coarsening acc and b/w vs coarsening ratio " )
    parser.add_argument('--seed', type= int,  default= 0, help = "PLotting a b/w vs coarsening acc and b/w vs coarsening ratio ")
    parser.add_argument('--adv_att', type = bool, default= False, help = "attack")
    parser.add_argument('--ptbr', type = float , default= 0, help = "attack")
    args = parser.parse_args()
    return args


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def fix_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def split(data, num_classes):
    indices = []
    num_test = (int)(data.num_nodes * 0.1 / num_classes)
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)
        test_index = torch.cat([i[:num_test] for i in indices], dim=0)
        val_index = torch.cat([i[num_test:num_test*2] for i in indices], dim=0)
        train_index = torch.cat([i[num_test*2:] for i in indices], dim=0)
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(test_index, size=data.num_nodes)
    return data

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    fix_seeds(args.seed)
    
    log_format = '%(asctime)s  %(name)8s  %(levelname)5s  %(message)s'
    logging.basicConfig(filename='Output.log',level=logging.INFO,format=log_format,filemode='a')
    logger = logging.getLogger("main")
    lr = []
    decay = []
    alpha = [1]
    beta = [1]
    if (args.lr >= 0):
        lr = [args.lr]
    if (args.decay >= 0):
        decay = [args.decay]
    if(args.data_choice==0):
        name_str = 'Planetoid Cora'
        if(args.normalize):
            dataset = Planetoid(root = 'data/Cora', name = 'Cora', transform=NormalizeFeatures())
            data = dataset[0] 
        else:
            dataset = Planetoid(root = 'data/Cora', name = 'Cora')
            data = dataset[0]
            print("not normalised")
        logger.info('planetoid')

    elif(args.data_choice==1):
        name_str = 'Planetoid Citeseer'
        if(args.normalize):
            dataset = Planetoid(root = 'data/CiteSeer', name = 'CiteSeer', transform=NormalizeFeatures())
            data = dataset[0] 
        else:
            dataset = Planetoid(root = 'data/CiteSeer', name = 'CiteSeer')
            data = dataset[0]

    elif(args.data_choice==2):
        name_str = 'Planetoid PubMed'
        if(args.normalize):
            dataset = Planetoid(root = 'data/PubMed', name = 'PubMed', transform=NormalizeFeatures())
            data = dataset[0] 
        else:
            dataset = Planetoid(root = 'data/PubMed', name = 'PubMed')
            data = dataset[0]

    elif(args.data_choice==3):
        name_str = 'CitationFull Cora'
        if(args.normalize):
            dataset = CitationFull(root = 'data/CitationCora', name = 'Cora', transform=NormalizeFeatures())
            data = dataset[0] 
        else:
            dataset = CitationFull(root = 'data/CitationCora', name = 'Cora')
            data = dataset[0]
        data = split(data,70)

    elif(args.data_choice==4):
        name_str = 'CitationFull CiteSeer'
        if(args.normalize):
            dataset = CitationFull(root = 'data/CitationCiteSeer', name = 'CiteSeer', transform=NormalizeFeatures())
            data = dataset[0] 
        else:
            dataset = CitationFull(root = 'data/CitationCiteSeer', name = 'CiteSeer')
            data = dataset[0]
        data = split(data,6)

    elif(args.data_choice==5):
        name_str = 'CitationFull DBLP'
        if(args.normalize):
            dataset = CitationFull(root = 'data/DBLP', name = 'DBLP', transform=NormalizeFeatures())
            data = dataset[0] 
        else:
            dataset = CitationFull(root = 'data/DBLP', name = 'DBLP')
            data = dataset[0]
        data = split(data,4)
        
    elif(args.data_choice==6):
        name_str = 'Coauthor CS'
        if(args.normalize):
            dataset = Coauthor(root = 'data/CS', name = 'CS', transform=NormalizeFeatures())
            data = dataset[0] 
        else:
            dataset = Coauthor(root = 'data/CS', name = 'CS')
            data = dataset[0]
        data = split(data,15)

    elif(args.data_choice==7):
        name_str = 'Coauthor Physics'
        if(args.normalize):
            dataset = Coauthor(root = 'data/Physics', name = 'Physics', transform=NormalizeFeatures())
            data = dataset[0] 
        else:
            dataset = Coauthor(root = 'data/Physics', name = 'Physics')
            data = dataset[0]
        data = split(data,5)

    elif(args.data_choice==8):
        name_str = 'Amazon computers'
        if(args.normalize):
            dataset = Amazon(root = 'data/Amazon_computers', name = 'computers', transform=NormalizeFeatures())
            data = dataset[0] 
        else:
            dataset = Amazon(root = 'data/Amazon_computers', name = 'computers')
            data = dataset[0]
        data = split(data,10)

    elif(args.data_choice==9):
        name_str = 'Amazon Photo'
        if(args.normalize):
            dataset = Amazon(root = 'data/Amazon_Photo', name = 'photo', transform=NormalizeFeatures())
            data = dataset[0] 
        else:
            dataset = Amazon(root = 'data/Amazon_Photo', name = 'photo')
            data = dataset[0]
        data = split(data,8)

    elif(args.data_choice==10):
        name_str = 'Reddit 2'
        if(args.normalize):
            dataset = Reddit2(root = 'data/Reddit2', transform=NormalizeFeatures())
            data = dataset[0] 
        else:
            dataset = Reddit2(root = 'data/Reddit2',allow_pickle=True)
            data = dataset[0]
        data = split(data,41)

    elif(args.data_choice==11):
        name_str = 'Reddit'
        if(args.normalize):
            dataset = Reddit(root = 'data/Reddit',transform=NormalizeFeatures())
            data = dataset[0] 
        else:
            dataset = Reddit(root = 'data/Reddit')
            data = dataset[0]
        data = split(data,41)

    elif(args.data_choice==12):
        name_str = 'Flickr'
        if(args.normalize):
            dataset = Flickr(root = 'data/Flickr',transform=NormalizeFeatures(),allow_pickle=True)
            data = dataset[0] 
        else:
            dataset = Flickr(root = 'data/Flickr',allow_pickle=True)
            data = dataset[0]
        data = split(data,7)
    
    elif(args.data_choice==13):
        raise NotImplementedError("Stochastic Block Model Not Implemented Yet")
    
    #lr = [30 , 3, 0.3, 0.03, 0.003, 0.0003, 0.00003]
    if (len(lr) == 0):
        lr = [0.03]
    #decay = [5e-01 ,5e-02 , 5e-03 , 5e-04 ,5e-05 , 5e-06, 5e-07]
    if (len(decay) == 0):
        decay = [5e-04]
    if (args.adv_att):
        print("attack done")
    if (args.plot):
        acc_arr = []
        bw_arr = []
        rr_arr = []
        bw_start = 0.9
        while (len(rr_arr) == 0 or rr_arr[-1] > 0.1 ):
            bw_arr.append(bw_start)
            out = run(dataset,data,bw_arr[-1],name_str,args.epochs, lr[0] , decay[0])
            acc_arr.append(out[0])
            rr_arr.append(out[1])
            dec = pow(10,math.floor(math.log10(bw_start)))
            if (abs(dec - bw_start) < bw_start/10):
                bw_start = 9* (bw_start/10)
            else:
                bw_start = bw_start - dec
        print(bw_arr)
        plt.plot(bw_arr, acc_arr,'-ok')
        # naming the x axis
        plt.xlabel('Bin Witdh Arrays')
        plt.xscale("log")
        # naming the y axis
        plt.ylabel('Accuracy on Test Set')
        # function to show the plot
        plt.savefig(name_str + "accuracy")
        plt.show()
        plt.plot(bw_arr, rr_arr,'-ok')
        # naming the x axis
        plt.xlabel('Bin Witdh Arrays')
        plt.xscale("log")
        # naming the y axis
        plt.ylabel('Reduction Ratio')
        # function to show the plot
        plt.savefig(name_str + "reduction_ratio")
        plt.show()
        plt.plot(rr_arr, acc_arr,'-ok')
        # naming the x axis
        plt.xlabel('reduction ratio')
        # naming the y axis
        plt.ylabel('Accuracy on Test Set')
        # function to show the plot
        plt.savefig(name_str + "rr_vs_accuracy")
        plt.show()


    elif(args.adv_att):
        print("Final Reported acc for reduction bw " + str(args.bw))
        print("For " + str(args.ptbr) + "% perturbation ")
        perturbed_data = PrePtbDataset(root='/tmp/',
                            name='cora',
                            attack_method='meta',
                            ptb_rate=args.ptbr)
        data = Dataset(root='/tmp/',
                            name='cora',
                            setting='prognn')
        n = perturbed_data.adj.shape[0]
        adj, features, labels = data.adj, data.features, data.labels
        perturbed_adj = perturbed_data.adj

        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        perturbed_adj, features, labels = preprocess(perturbed_adj, features, labels, preprocess_adj=False, sparse= False, device= 'cpu')
        perturbed_adj = torch.FloatTensor(((sparse_mx_to_torch_sparse_tensor((perturbed_data.adj)))).to_dense())
        from torch_geometric.data import Data
        edges_src = (torch.nonzero( perturbed_adj, as_tuple=True))[0].long()
        edges_dst = (torch.nonzero( perturbed_adj, as_tuple=True))[1].long()
        edge_index_corsen = torch.stack((edges_src, edges_dst))
        data = Data(x= features, edge_index = edge_index_corsen, y = labels)
        data.train_mask = torch.zeros((n,), dtype = bool)
        data.train_mask[idx_train] = True
        data.val_mask = torch.zeros((n,), dtype = bool)
        data.val_mask[idx_val] = True
        data.test_mask = torch.zeros((n,), dtype = bool)
        data.test_mask[idx_test] = True
        acc_arr = []
        rr_arr = []
        print("Starting method on " + name_str + " for bin-width " + str(args.bw) + " and gsp refining " + str(args.gsp) + " on " + str(device))
        for l in lr:
            for d in decay:
                for a in alpha:
                    for b in beta:
                        for c in range(0,args.reps):
                            out = run(dataset,data,args.bw,name_str,args.epochs, l , d, args.gsp, a, b)
                            acc_arr.append(out[0])
                            rr_arr.append(out[1])
                        mean_acc = np.mean(np.array(acc_arr))
                        std_acc = np.std(np.array(acc_arr))
                        mean_rr = np.mean(np.array(rr_arr))
                        std_rr = np.std(np.array(rr_arr))
                        print("Final Reported acc for reduction " + str(mean_rr) + " +- "+ str(std_rr) + " learning rate " + str(l) + " decay " + str(d) + " alpha "+ str(a) + " and beta " + str(b) + " is " + str(mean_acc) + " +- " + str(std_acc))
                        logger.info("Final Reported acc for learning rate " + str(l) + " and decay " + str(d) + " is " + str(mean_acc) + " +- " + str(std_acc))
                        acc_arr.clear()
                        rr_arr.clear()
        print("Done!")
        logger.info("Done!")

    else:
        if (args.tuning_gcn):
            lr = [30 , 3, 0.3, 0.03, 0.003, 0.0003, 0.00003]
            decay = [5e-01 ,5e-02 , 5e-03 , 5e-04 ,5e-05 , 5e-06, 5e-07]
        if (args.tuning_opt):
            alpha = [100,10,1,0.1,0.01]
            beta = [100,10,1,0.1,0.01]
        acc_arr = []
        rr_arr = []
        print("Starting method on " + name_str + " for bin-width " + str(args.bw) + " and gsp refining " + str(args.gsp) + " on " + str(device))
        for l in lr:
            for d in decay:
                for a in alpha:
                    for b in beta:
                        for c in range(0,args.reps):
                            out = run(dataset,data,args.bw,name_str,args.epochs, l , d, args.gsp, a, b)
                            acc_arr.append(out[0])
                            rr_arr.append(out[1])
                        mean_acc = np.mean(np.array(acc_arr))
                        std_acc = np.std(np.array(acc_arr))
                        mean_rr = np.mean(np.array(rr_arr))
                        std_rr = np.std(np.array(rr_arr))
                        print("Final Reported acc for reduction " + str(mean_rr) + " +- "+ str(std_rr) + " learning rate " + str(l) + " decay " + str(d) + " alpha "+ str(a) + " and beta " + str(b) + " is " + str(mean_acc) + " +- " + str(std_acc))
                        logger.info("Final Reported acc for learning rate " + str(l) + " and decay " + str(d) + " is " + str(mean_acc) + " +- " + str(std_acc))
                        acc_arr.clear()
                        rr_arr.clear()
        print("Done!")
        logger.info("Done!")


    


        

        

