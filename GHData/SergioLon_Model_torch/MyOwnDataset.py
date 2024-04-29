import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import glob
import pyvista as pv
import numpy as np
#import trimesh as tr

#import matplotlib.pyplot as plt
import torch

from torch_geometric.transforms import FaceToEdge,Compose,GenerateMeshNormals
from new_random_rotate import RandomRotate
from torch_geometric.data import Data,DataLoader,InMemoryDataset

class Normilize_WSS(object):
    def __call__(self,data):
        # maxm = data.wss.max()
        # minm = data.wss.min()
        # print("OLD WSS MAX: ",data.wss.max())
        # print("OLD WSS MIN: ",data.wss.min())
        # #wss_mean = ( maxm + minm ) / 2.
        # wss_semidisp = ( maxm - minm )
        # data.wss=(data.wss-minm)/wss_semidisp
        # print("NEW WSS MAX: ",data.wss.max())
        # print("NEW WSS MIN: ",data.wss.min())
        
        # print("OLD WSS_X MAX: ",data.wss_x.max())
        # print("OLD WSS_X MIN: ",data.wss_x.min())
        # maxm_x = data.wss_x.max()
        # minm_x = data.wss_x.min()
        # wss_x_semidisp = ( maxm_x- minm_x )
        # data.wss_x=(data.wss_x-minm_x)/wss_x_semidisp
        # print("NEW WSS_X MAX: ",data.wss_x.max())
        # print("NEW WSS_X MIN: ",data.wss_x.min())
        # maxm_y = data.wss_y.max()
        # minm_y = data.wss_y.min()
        # wss_y_semidisp = ( maxm_y- minm_y )
        # data.wss_y=(data.wss_y-minm_y)/wss_y_semidisp
        
        # maxm_z = data.wss_z.max()
        # minm_z = data.wss_z.min()
        # wss_z_semidisp = ( maxm_z- minm_z )
        # data.wss_z=(data.wss_z-minm_z)/wss_z_semidisp
        
        # maxm_abs = data.wss_abs.max()
        # minm_abs = data.wss_abs.min()
        # wss_abs_semidisp = ( maxm_abs- minm_abs )
        # data.wss_abs=(data.wss_abs-minm_abs)/wss_abs_semidisp
        maxm = data.wss_coord.max(dim=-2).values
        minm = data.wss_coord.min(dim=-2).values
        data.wss_max[:]=maxm.max()
        data.wss_min[:]=minm.min()
        print("OLD WSS MAX: ",maxm)
        print("OLD WSS MIN: ",minm)
        # print("OLD POS_X MAX: ",data.pos_x.max())
        # print("OLD POS_X MIN: ",data.pos_x.min())
        mean = ( maxm.max() + minm.min() ) / 2.
        data.wss_coord = (data.wss_coord - minm.min()) / ( (maxm.max() - minm.min()))
        #data.wss_coord = (data.wss_coord - mean) / ( (maxm.max() - minm.min())/2.)
        #data.pos_x=((data.pos_x - minm[0]) / ( (maxm[0] - minm[0])))
        #data.pos_y=torch.tensor(np.expand_dims(data.pos[:,1].detach().numpy(),axis=-1))
        #data.pos_z=torch.tensor(np.expand_dims(data.pos[:,2].detach().numpy(),axis=-1))
        #print(data.pos_x.size())
        print("NEW WSS MAX: ",data.wss_coord.max(dim=-2).values)
        print("NEW WSS MIN: ",data.wss_coord.min(dim=-2).values)
        return data
    
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class Normalize_vertx(object):
    def __call__(self,data):
        maxm = data.pos.max(dim=-2).values
        minm = data.pos.min(dim=-2).values
        #data.vrtx_max[0,:]=maxm[:]
        #data.vrtx_min[0,:]=minm[:]
        print("OLD VERTX MAX: ",maxm)
        print("OLD VERTX MIN: ",minm)
        # print("OLD POS_X MAX: ",data.pos_x.max())
        # print("OLD POS_X MIN: ",data.pos_x.min())
        #mean = ( maxm + minm ) / 2.
        #data.pos = (data.pos - mean) / ( (maxm - minm)/2)
        data.pos =data.pos/maxm.max()
        #data.pos_x=((data.pos_x - minm[0]) / ( (maxm[0] - minm[0])))
        #data.pos_y=torch.tensor(np.expand_dims(data.pos[:,1].detach().numpy(),axis=-1))
        #data.pos_z=torch.tensor(np.expand_dims(data.pos[:,2].detach().numpy(),axis=-1))
        #print(data.pos_x.size())
        print("NEW VRTX MAX: ",data.pos.max(dim=-2).values)
        print("NEW VRTX MIN: ",data.pos.min(dim=-2).values)
        
        # print("NEW POS_X MAX: ",data.pos_x.max())
        # print("NEW POS_X MIN: ",data.pos_x.min())
        return data
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
    
p_trans= [Normalize_vertx(),Normilize_WSS(),]

pre_trans=Compose(p_trans)

trans=[
       RandomRotate(90,axis=0),
       RandomRotate(90,axis=2),
       RandomRotate(90,axis=1),
       ]

pos_trans=Compose(trans)

class MyOwnDataset(InMemoryDataset):
     def __init__(self, 
                  root, 
                  #transform=pos_trans,
                  transform=None,
                  pre_transform=pre_trans):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        #self.transform=transform
        self.pre_transform=pre_transform

     @property
     def raw_file_names(self):
         file_name=glob.glob(os.path.join(self.root,'*.vtp'))
         return file_name

     @property
     def processed_file_names(self):
        return ['data.pt']


     def process(self):
        # Read data into huge `Data` list.
        data_list = []
        f2e=FaceToEdge(remove_faces=(False))
        norm=GenerateMeshNormals()
        for ii,name in enumerate(self.raw_file_names):
            # print(name)
            mesh=pv.read(name)
            #mesh=mesh.connectivity()
            #mesh.save('../Meshes_vtp/torch_dataset_xyz/raw/Decimated/'+str(ii)+'.vtp')
            # if int(''.join(filter(str.isdigit, name)))==16:
            #     #print("TRUE")
            #     mesh.point_arrays["WSS magnitude"]/=1e3
            # if int(''.join(filter(str.isdigit, name)))==18:
            #     #print("TRUE")
            #     mesh.point_arrays["WSS magnitude"]/=1e2
            
            faces=mesh.faces.reshape((-1,4))[:, 1:4].T
            pos=torch.tensor(mesh.points,dtype=torch.float)
            vrtx_max=pos.max(dim=-2).values
            vrtx_min=pos.min(dim=-2).values
            pos=pos-((vrtx_max-vrtx_min)/2)
            # print("VERTX MAX PRE TRANSL: ",vrtx_max)
            # print("VERTX MIN PRE TRANSL: ",vrtx_min)
            # vrtx_max=pos.max(dim=-2).values
            # vrtx_min=pos.min(dim=-2).values
            # print("VERTX MAX POST TRANSL: ",vrtx_max)
            # print("VERTX MIN POST TRANSL: ",vrtx_min)
            #pos_x=torch.tensor(np.expand_dims(mesh.points[:,0],axis=-1))
            # pos_y=torch.tensor(np.expand_dims(mesh.points[:,1],axis=-1))
            # pos_z=torch.tensor(np.expand_dims(mesh.points[:,2],axis=-1))
            #print(pos.size())
            faces=torch.LongTensor(faces)
            #wss=torch.tensor(np.expand_dims(mesh.point_arrays["WSS magnitude"],axis=-1),dtype=torch.float)
            wss_x=torch.tensor(np.expand_dims(mesh.point_arrays["wss_x"],axis=-1),dtype=torch.float)
            wss_y=torch.tensor(np.expand_dims(mesh.point_arrays["wss_y"],axis=-1),dtype=torch.float)
            wss_z=torch.tensor(np.expand_dims(mesh.point_arrays["wss_z"],axis=-1),dtype=torch.float)
            
            #print(wss.size())
            wss_abs=torch.tensor(np.expand_dims(mesh.point_arrays["wss_abs"],axis=-1),dtype=torch.float)
            wss_coord=torch.cat([wss_x,wss_y,wss_z],dim=1)
            # wss_max=torch.zeros((1,4))
            # wss_min=torch.zeros((1,4))
            # vrtx_max=torch.zeros((1,3))
            # vrtx_min=torch.zeros((1,3))
            #print(wss.size(0))
            ##
            # prova=np.zeros((pos.size(0),1))
            # prova[:int(pos.size(0)/2)]=0.5
            
            # prova=torch.tensor(prova)
            
            ##
            data=Data(
                pos=pos,
                #pos_x=pos_x,
                # pos_y=None,
                # pos_z=None,
                face=faces,
                #wss=wss,
                # wss_x=wss_x,
                # wss_y=wss_y,
                # wss_z=wss_z,
                wss_coord=wss_coord,
                wss_abs=wss_abs,
                wss_max=0,
                wss_min=0,
                # vrtx_max=vrtx_max,
                # vrtx_min=vrtx_min,
                # wss_abs=wss_abs,
                #prova=prova,
                )
            
            data=f2e(data)
            #data_aug=pos_trans(data)
            data=norm(data)
            #data_aug=norm(data_aug)
            #print(data)
            data_list.append(data)
            
            #data_list.append(data_aug)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            #data_list = [self.pre_transform(data) for data in data_list]
            data, slices = self.collate(data_list)
            data=self.pre_transform(data)
        else:
            data, slices = self.collate(data_list)
        
        torch.save((data, slices), self.processed_paths[0])


dataset=MyOwnDataset(root='../Meshes_vtp/torch_dataset_xyz/raw/New_Decimated',)

# for b in DataLoader(dataset,batch_size=1):
#     print(b.pos)
