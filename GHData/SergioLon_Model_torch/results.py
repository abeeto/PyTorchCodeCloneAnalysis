import torch 
import matplotlib.pyplot as plt
#from split_dataset import split_dataset
import numpy as np
#from model import GCN
import pyvista as pv
from torch_geometric.transforms import FaceToEdge,GenerateMeshNormals
from torch_geometric.data import Data,DataLoader 
from MyOwnDataset import MyOwnDataset
from losses import nmse
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


"""
This function use the model given to make a prediction on a mesh,
'known' is a bool, if True it means that the  original wss values are known, so
the prediction and the original can be compared
"""
def apply_model_on_mesh(my_path,model,device,data_loaders_training,known=True):
    
    #mesh=pv.read(file_name)
    #preprocess the input
    # faces=mesh.faces.reshape((-1,4))[:, 1:4].T
    # pos=torch.tensor(mesh.points,dtype=torch.float)
    
    # faces=torch.LongTensor(faces)
    # # mean = ( vrtx_maxm + vrtx_minm ) / 2.
    
    # data=Data(
    #                 pos=pos,
    #                 face=faces,
    #         )
    
    # f2e=FaceToEdge(remove_faces=(False))
    # norm=GenerateMeshNormals()
    
    # data=f2e(data)
    # data=norm(data)
    # data=data.to(device)
    # mean = ( vrtx_maxm + vrtx_minm ) / 2.
    # data.pos = (data.pos - mean) / ( (vrtx_maxm - vrtx_minm)/2)
    dataset = MyOwnDataset(my_path)
    
    dataset.data=dataset.data.to(device)
    loaders = DataLoader(dataset, batch_size=1)
    data_loaders={'val':loaders}
    predict_on_dataloader(model,data_loaders,data_loaders_training)
    
    
def predict_on_dataloader(model,data_loaders,data_loaders_training=None):
    model.eval()
    # for idx,m in enumerate(data_loaders['train']):
    #     if m.wss_max[0,0]!=0:
    #         wss_maxm=m.wss_max
    #         wss_minm=m.wss_min
    #         vrtx_maxm=m.vrtx_max
    #         vrtx_minm=m.vrtx_min
            
    for idx,m in enumerate(data_loaders['val']):
        
        if data_loaders_training is not None:
            for ii,t in enumerate(data_loaders_training['val']):
                print("Denormalize and renormalize WSS Procedure")
                m.wss_coord =( m.wss_coord*(m.wss_max - m.wss_min))+m.wss_min
                m.wss_max=t.wss_max
                m.wss_min=t.wss_min
                print("MAX TO UNNORMALIZE: ",t.wss_max)
                print("MIN TO UNNORMALIZE: ",t.wss_min)
                m.wss_coord = (m.wss_coord - m.wss_min) / (m.wss_max - m.wss_min)
                break
        if idx==0:
            # if m.wss_max[0,0]!=0:
            #     wss_maxm=m.wss_max
            #     wss_minm=m.wss_min
            #     vrtx_maxm=m.vrtx_max
            #     vrtx_minm=m.vrtx_min
            
            out=model(m)
            print("NMSE: ",nmse(out, m.wss_coord).cpu().detach().numpy())
            # a=torch.sqrt(out[:,0]**2+out[:,1]**2+out[:,2]**2).unsqueeze(1)
            # fig, ax = plt.subplots()
            # ax.plot(m.wss[:,3].cpu(),label='Real')
            # ax.plot(a.cpu().detach().numpy(),label='Pred')
            # ax.legend()
            # #ax.title('One Val sample')
            # ax.set_xlabel('Vertx')
            # ax.set_ylabel('WSS_ABS normalized')
            plt.show()
            fig, ax = plt.subplots()
            ax.plot(m.wss_coord[:,0].cpu(),label='Real')
            ax.plot(out[:,0].cpu().detach().numpy(),label='Pred')
            ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('WSS_X normalized')
            plt.show()
            fig, ax = plt.subplots()
            ax.plot(m.wss_coord[:,1].cpu(),label='Real')
            ax.plot(out[:,1].cpu().detach().numpy(),label='Pred')
            ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('WSS_Y normalized')
            plt.show()
            fig, ax = plt.subplots()
            ax.plot(m.wss_coord[:,2].cpu(),label='Real')
            ax.plot(out[:,2].cpu().detach().numpy(),label='Pred')
            ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('WSS_Z normalized')
            plt.show()
            
            #creating the predicted mesh
            data=m.to('cpu')
            nodes=data.pos.numpy()
            cells=data.face.numpy()
            temp=np.array([3]*cells.shape[1])
            cells=np.c_[temp,cells.T].ravel()
            mesh=pv.PolyData(nodes,cells)
            # print("MAX: ",m.wss_max)
            # print("MIN: ",m.wss_min)
            wss_x=denormalize_wss(out[:,0].cpu().detach().numpy(),m.wss_max.cpu(),m.wss_min.cpu())
            wss_y=denormalize_wss(out[:,1].cpu().detach().numpy(),m.wss_max.cpu(),m.wss_min.cpu())
            wss_z=denormalize_wss(out[:,2].cpu().detach().numpy(),m.wss_max.cpu(),m.wss_min.cpu())
            
            mesh.point_arrays["wss_x_pred"]=wss_x
            mesh.point_arrays["wss_y_pred"]=wss_y
            mesh.point_arrays["wss_z_pred"]=wss_z
            mesh.point_arrays["wss_abs_pred"]=np.expand_dims(np.sqrt(wss_x**2+wss_y**2+wss_z**2),axis=-1)
            mesh.point_arrays["wss_x"]=denormalize_wss(m.wss_coord[:,0].cpu().detach().numpy(),m.wss_max.cpu(),m.wss_min.cpu())
            mesh.point_arrays["wss_y"]=denormalize_wss(m.wss_coord[:,1].cpu().detach().numpy(),m.wss_max.cpu(),m.wss_min.cpu())
            mesh.point_arrays["wss_z"]=denormalize_wss(m.wss_coord[:,2].cpu().detach().numpy(),m.wss_max.cpu(),m.wss_min.cpu())
            mesh.point_arrays["wss_abs"]=m.wss_abs.cpu().detach().numpy()
            
            ##
            value = input("Choose a name for the prediction file:\n")
            out_name='../Meshes_vtp/torch_dataset_xyz/raw/New_Decimated/Predicted/'+value+'.vtp'
            
            
            ##
             # X COMPONENT
            fig, ax = plt.subplots()
            
            ax.plot(np.abs(mesh.point_arrays["wss_x_pred"]-mesh.point_arrays["wss_x"]))
            #ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('|WSS_X-WSS_X_PRE|')
            plt.show()
            # Y COMPONENT
            fig, ax = plt.subplots()
            
            ax.plot(np.abs(mesh.point_arrays["wss_y_pred"]-mesh.point_arrays["wss_y"]))
            #ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('|WSS_Y-WSS_Y_PRED|')
            plt.show()
            #Z COMPONENT
            fig, ax = plt.subplots()
            
            ax.plot(np.abs(mesh.point_arrays["wss_z_pred"]-mesh.point_arrays["wss_z"]))
            #ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('|WSS_Z-WSS_Z_PRED|')
            plt.show()
            #ABS
            fig, ax = plt.subplots()
            
            ax.plot(np.abs(mesh.point_arrays["wss_abs_pred"]-mesh.point_arrays["wss_abs"]))
            #ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('|WSS_ABS-WSS_ABS_PRED|')
            plt.show()
            #ERRORE PERCHENTUALE
            # X COMPONENT
            fig, ax = plt.subplots()
            #err_x=np.abs((mesh.point_arrays["wss_x_pred"]-mesh.point_arrays["wss_x"])/mesh.point_arrays["wss_x"])*100
            err_x=np.zeros((mesh.point_arrays["wss_x_pred"].shape[0],1))
            for i in range(err_x.shape[0]):
                err_x[i]=abs((mesh.point_arrays["wss_x_pred"][i]-mesh.point_arrays["wss_x"][i]))/max(abs(mesh.point_arrays["wss_x"]))
            #print("err_x.size(0)")
            ax.plot(err_x)
            #ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('% Error WSS_X')
            plt.show()
            # Y COMPONENT
            fig, ax = plt.subplots()
            err_y=np.abs((mesh.point_arrays["wss_y_pred"]-mesh.point_arrays["wss_y"]))/max(abs(mesh.point_arrays["wss_y"]))
            ax.plot(err_y)
            #ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('% Error WSS_Y')
            plt.show()
            #Z COMPONENT
            fig, ax = plt.subplots()
            err_z=np.abs((mesh.point_arrays["wss_z_pred"]-mesh.point_arrays["wss_z"]))/max(abs(mesh.point_arrays["wss_z"]))
            ax.plot(err_z)
            #ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('% Error WSS_Z')
            plt.show()
            #ABS
            fig, ax = plt.subplots()
            err_abs=np.abs((mesh.point_arrays["wss_abs_pred"]-mesh.point_arrays["wss_abs"]))/max(abs(mesh.point_arrays["wss_abs"]))
            ax.plot(err_abs)
            #ax.legend()
            #ax.title('One Val sample')
            ax.set_xlabel('Vertx')
            ax.set_ylabel('% Error WSS_ABS')
            plt.show()
            
            mesh.point_arrays["err_z"]=err_z
            mesh.point_arrays["err_x"]=err_x
            mesh.point_arrays["err_y"]=err_y
            mesh.point_arrays["err_abs"]=err_abs
            mesh.save(out_name)
            break