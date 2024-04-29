import torch 
import matplotlib.pyplot as plt
from split_dataset import split_dataset
import numpy as np
from model import GCN
import pyvista as pv
from torch_geometric.transforms import FaceToEdge,RandomRotate,Compose,GenerateMeshNormals
from torch_geometric.data import Data,DataLoader,InMemoryDataset
from losses import nmse
from torch.optim.lr_scheduler import ReduceLROnPlateau


def denormalize_wss(point_array,maxm,minm):
    #maxm=point_array.max()
    #minm=point_array.min()
    # print("OLD MAX: ",maxm)
    # print("OLD MIN: ",minm)
    #print(maxm)
    maxm=maxm[0].detach().numpy()
    minm=minm[0].detach().numpy()
    
    new_array=((point_array)*(maxm-minm))+minm
    # print("NEW MAX: ",new_array.max())
    # print("NEW MIN: ",new_array.min())
    return new_array
#%% SETTING PARAMS
meshes_path='../Meshes_vtp/torch_dataset_xyz/raw/New_Decimated'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device='cpu'
#dataset=my_train_fn()
#loader=DataLoader(dataset,batch_size=1)
#dataset=MyOwnDataset(root='../Meshes_vtp',)
hyperParams={
    "lr": 0.001,
    "epochs": 1000,
    "batch_size":1,
    "val_split":0.1,
    "loss":torch.nn.MSELoss(),
    "weight_decay":5e-6
    
    }

#WEIGHT INITIALIZATION 
def weights_init_uniform_rule(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('FeaStConv') != -1:
            # get the number of the inputs
            n = m.in_channels
            y = 1.0/np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            #m.bias.data.fill_(0)
        if classname.find('Linear') != -1:
            # get the number of the inputs
            n = m.in_features
            y = 1.0/np.sqrt(n)
            m.weight.data.uniform_(-y, y)
    # create a new model with these weights
model = GCN()
#model.apply(weights_init_uniform_rule)
#%% SETTING FOR TRAINING

#loader=DataLoader(dataset,batch_size=1)
data_loaders=split_dataset(device,
                           meshes_path, 
                           hyperParams['batch_size'], 
                           hyperParams['val_split']
                           )
print('-'*40+'\nDATASET CREATED')



optimizer = torch.optim.Adam(model.parameters(), 
                             lr=hyperParams['lr'], 
                             weight_decay=hyperParams['weight_decay']
                             )

scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=100,
        verbose=True
    )
criterion = hyperParams['loss']

model=model.to(device)  
saved_loss=np.zeros((2,hyperParams["epochs"]),dtype=np.dtype('float32'))


#%% TRAINING
#norm=GenerateMeshNormals()
print('-'*40+'\nTRAINING  STARTED\n'+'-'*40)
for epoch in range(hyperParams['epochs']):
    try:
        
        print(f'Epoch: {epoch:3d}')
        train_loss=0.0
        val_loss=0.0
        for phase in ['train', 'val']:
            if phase == 'train':
                
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
                
            
            #running_loss = 0.0
            for ii,batch in enumerate(data_loaders[phase]):
                # if ii==0 and phase=='val':
                #     print(batch.vrtx_max)
                #     #print(batch.wss_max)
                #     maxm=batch.wss_max
                #     minm=batch.wss_min
                    
                #print(batch.pos.size(0))
                #batch=norm(batch)
                out = model(batch)  # Perform a single forward pass.
                #print(ii)
                
                loss = nmse(out, batch.wss_coord)  # Compute the loss solely based on the training nodes.
                #loss_x = nmse(out[:,0], batch.wss[:,0])
                #loss_abs = nmse(out[:,1], batch.wss[:,3])
                #loss=loss_x+loss_abs
                optimizer.zero_grad()
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()
                    train_loss+=loss.data
                else:
                    val_loss+=loss.data
                    scheduler.step(val_loss)
    
        
        train_loss /= len(data_loaders['train'])
        val_loss /=len(data_loaders['val'])
        #save the loss
        saved_loss[0][epoch]=train_loss
        saved_loss[1][epoch]=val_loss
        #print loss for each epoch
        print('{} Loss: {:.4f}; {} Loss: {:.4f}'.format('Train', train_loss,'Val',val_loss))
    except KeyboardInterrupt:
        break
#%% LOSS PLOT
fig, ax = plt.subplots()
ax.plot(range(hyperParams['epochs']),saved_loss[0],label='Train')
ax.plot(range(hyperParams['epochs']),saved_loss[1],label='Val')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
plt.yscale("log")
plt.show()

#%%
from results import apply_model_on_mesh,predict_on_dataloader

#wss_maxm,wss_minm,vrtx_maxm,vrtx_minm=predict_on_dataloader(model,data_loaders)
predict_on_dataloader(model,data_loaders,data_loaders_training=None)
#%%
# #%%
# file_name='../Meshes_vtp/torch_dataset_xyz/raw/New_Decimated/Clipped/aorta_0_dec_cl2.vtp'
# # out_name='../Meshes_vtp/torch_dataset_xyz/raw/New_Decimated/Predicted/aorta_0_pred.vtp'
# value = input("Do you want to make a prediction on "+file_name+"? [y/n]\n")
# if value=='y':
#     value = input("Choose a name for the prediction file:\n")
    
#     out_name='../Meshes_vtp/torch_dataset_xyz/raw/New_Decimated/Predicted/'+value+'.vtp'
my_path='../Meshes_vtp/torch_dataset_xyz/raw/New_Decimated/Caso sano'
apply_model_on_mesh(my_path,model,device,data_loaders,known=True)

  
#%%
# model.eval()
# # for idx,m in enumerate(data_loaders['train']):
# #     if m.wss_max[0,0]!=0:
# #         wss_maxm=m.wss_max
# #         wss_minm=m.wss_min
# #         vrtx_maxm=m.vrtx_max
# #         vrtx_minm=m.vrtx_min
        
# for idx,m in enumerate(data_loaders['val']):
#     if idx==0:
#         # if m.wss_max[0,0]!=0:
#         #     wss_maxm=m.wss_max
#         #     wss_minm=m.wss_min
#         #     vrtx_maxm=m.vrtx_max
#         #     vrtx_minm=m.vrtx_min
        
#         out=model(m)
#         # a=torch.sqrt(out[:,0]**2+out[:,1]**2+out[:,2]**2).unsqueeze(1)
#         # fig, ax = plt.subplots()
#         # ax.plot(m.wss[:,3].cpu(),label='Real')
#         # ax.plot(a.cpu().detach().numpy(),label='Pred')
#         # ax.legend()
#         # #ax.title('One Val sample')
#         # ax.set_xlabel('Vertx')
#         # ax.set_ylabel('WSS_ABS normalized')
#         plt.show()
#         fig, ax = plt.subplots()
#         ax.plot(m.wss_coord[:,0].cpu(),label='Real')
#         ax.plot(out[:,0].cpu().detach().numpy(),label='Pred')
#         ax.legend()
#         #ax.title('One Val sample')
#         ax.set_xlabel('Vertx')
#         ax.set_ylabel('WSS_X normalized')
#         plt.show()
#         fig, ax = plt.subplots()
#         ax.plot(m.wss_coord[:,1].cpu(),label='Real')
#         ax.plot(out[:,1].cpu().detach().numpy(),label='Pred')
#         ax.legend()
#         #ax.title('One Val sample')
#         ax.set_xlabel('Vertx')
#         ax.set_ylabel('WSS_Y normalized')
#         plt.show()
#         fig, ax = plt.subplots()
#         ax.plot(m.wss_coord[:,2].cpu(),label='Real')
#         ax.plot(out[:,2].cpu().detach().numpy(),label='Pred')
#         ax.legend()
#         #ax.title('One Val sample')
#         ax.set_xlabel('Vertx')
#         ax.set_ylabel('WSS_Z normalized')
#         plt.show()
        
#         #creating the predicted mesh
#         data=m.to('cpu')
#         nodes=data.pos.numpy()
#         cells=data.face.numpy()
#         temp=np.array([3]*cells.shape[1])
#         cells=np.c_[temp,cells.T].ravel()
#         mesh=pv.PolyData(nodes,cells)
        
#         wss_x=denormalize_wss(out[:,0].cpu().detach().numpy(),m.wss_max.cpu(),m.wss_min.cpu())
#         wss_y=denormalize_wss(out[:,1].cpu().detach().numpy(),m.wss_max.cpu(),m.wss_min.cpu())
#         wss_z=denormalize_wss(out[:,2].cpu().detach().numpy(),m.wss_max.cpu(),m.wss_min.cpu())
        
#         mesh.point_arrays["wss_x_pred"]=wss_x
#         mesh.point_arrays["wss_y_pred"]=wss_y
#         mesh.point_arrays["wss_z_pred"]=wss_z
#         mesh.point_arrays["wss_abs_pred"]=np.expand_dims(np.sqrt(wss_x**2+wss_y**2+wss_z**2),axis=-1)
#         mesh.point_arrays["wss_x"]=denormalize_wss(m.wss_coord[:,0].cpu().detach().numpy(),m.wss_max.cpu(),m.wss_min.cpu())
#         mesh.point_arrays["wss_y"]=denormalize_wss(m.wss_coord[:,1].cpu().detach().numpy(),m.wss_max.cpu(),m.wss_min.cpu())
#         mesh.point_arrays["wss_z"]=denormalize_wss(m.wss_coord[:,2].cpu().detach().numpy(),m.wss_max.cpu(),m.wss_min.cpu())
#         mesh.point_arrays["wss_abs"]=m.wss_abs.cpu().detach().numpy()
        
#         ##
#         value = input("Choose a name for the prediction file:\n")
#         out_name='../Meshes_vtp/torch_dataset_xyz/raw/New_Decimated/Predicted/'+value+'.vtp'
        
        
#         ##
#          # X COMPONENT
#         fig, ax = plt.subplots()
        
#         ax.plot(np.abs(mesh.point_arrays["wss_x_pred"]-mesh.point_arrays["wss_x"]))
#         #ax.legend()
#         #ax.title('One Val sample')
#         ax.set_xlabel('Vertx')
#         ax.set_ylabel('|WSS_X-WSS_X_PRE|')
#         plt.show()
#         # Y COMPONENT
#         fig, ax = plt.subplots()
        
#         ax.plot(np.abs(mesh.point_arrays["wss_y_pred"]-mesh.point_arrays["wss_y"]))
#         #ax.legend()
#         #ax.title('One Val sample')
#         ax.set_xlabel('Vertx')
#         ax.set_ylabel('|WSS_Y-WSS_Y_PRED|')
#         plt.show()
#         #Z COMPONENT
#         fig, ax = plt.subplots()
        
#         ax.plot(np.abs(mesh.point_arrays["wss_z_pred"]-mesh.point_arrays["wss_z"]))
#         #ax.legend()
#         #ax.title('One Val sample')
#         ax.set_xlabel('Vertx')
#         ax.set_ylabel('|WSS_Z-WSS_Z_PRED|')
#         plt.show()
#         #ABS
#         fig, ax = plt.subplots()
        
#         ax.plot(np.abs(mesh.point_arrays["wss_abs_pred"]-mesh.point_arrays["wss_abs"]))
#         #ax.legend()
#         #ax.title('One Val sample')
#         ax.set_xlabel('Vertx')
#         ax.set_ylabel('|WSS_ABS-WSS_ABS_PRED|')
#         plt.show()
#         #ERRORE PERCHENTUALE
#         # X COMPONENT
#         fig, ax = plt.subplots()
#         #err_x=np.abs((mesh.point_arrays["wss_x_pred"]-mesh.point_arrays["wss_x"])/mesh.point_arrays["wss_x"])*100
#         err_x=np.zeros((mesh.point_arrays["wss_x_pred"].shape[0],1))
#         for i in range(err_x.shape[0]):
#             err_x[i]=abs((mesh.point_arrays["wss_x_pred"][i]-mesh.point_arrays["wss_x"][i]))/max(abs(mesh.point_arrays["wss_x"]))
#         #print("err_x.size(0)")
#         ax.plot(err_x)
#         #ax.legend()
#         #ax.title('One Val sample')
#         ax.set_xlabel('Vertx')
#         ax.set_ylabel('% Error WSS_X')
#         plt.show()
#         # Y COMPONENT
#         fig, ax = plt.subplots()
#         err_y=np.abs((mesh.point_arrays["wss_y_pred"]-mesh.point_arrays["wss_y"]))/max(abs(mesh.point_arrays["wss_y"]))
#         ax.plot(err_y)
#         #ax.legend()
#         #ax.title('One Val sample')
#         ax.set_xlabel('Vertx')
#         ax.set_ylabel('% Error WSS_Y')
#         plt.show()
#         #Z COMPONENT
#         fig, ax = plt.subplots()
#         err_z=np.abs((mesh.point_arrays["wss_z_pred"]-mesh.point_arrays["wss_z"]))/max(abs(mesh.point_arrays["wss_z"]))
#         ax.plot(err_z)
#         #ax.legend()
#         #ax.title('One Val sample')
#         ax.set_xlabel('Vertx')
#         ax.set_ylabel('% Error WSS_Z')
#         plt.show()
#         #ABS
#         fig, ax = plt.subplots()
#         err_abs=np.abs((mesh.point_arrays["wss_abs_pred"]-mesh.point_arrays["wss_abs"]))/max(abs(mesh.point_arrays["wss_abs"]))
#         ax.plot(err_abs)
#         #ax.legend()
#         #ax.title('One Val sample')
#         ax.set_xlabel('Vertx')
#         ax.set_ylabel('% Error WSS_ABS')
#         plt.show()
        
#         mesh.point_arrays["err_z"]=err_z
#         mesh.point_arrays["err_x"]=err_x
#         mesh.point_arrays["err_y"]=err_y
#         mesh.point_arrays["err_abs"]=err_abs
#         mesh.save(out_name)
#         break