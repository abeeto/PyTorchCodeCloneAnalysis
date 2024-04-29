import numpy as np
import imageio

from prediction import *








class Eval_Obj():

    def __init__(self, num_classes):    
        self.num_classes=num_classes

        self.pred_list=[];  
        self.ground_list=[];
        self.viz_list=[];

        self.iou_list=[];
        self.class_iou_list=[];
        for i in range(num_classes+1):
            self.class_iou_list.append([])

        return



    def aggregate(self, my_ground, my_pred):
        for i in range(len(my_pred)):
            self.ground_list.append(my_ground[i])
            self.pred_list.append(my_pred[i])
        return



    def get_one_hot(self, im):

        im=im.astype(int)

        seg_out=np.zeros((im.shape[0], im.shape[1], self.num_classes+1))
        # Restructure seg map into one-hot encoded version
        for ii in range(im.shape[0]):
            for jj in range(im.shape[1]):
                seg_out[ii,jj,int(im[ii,jj])]=1

        return seg_out





    def eval(self):

        # Get total_IOU LIST
        for i in range(len(self.ground_list)):
            print(i)
            pred=self.pred_list[i].pxd
            pxd=self.ground_list[i].pxd     

            pred=self.get_one_hot(pred)
            pxd=self.get_one_hot(pxd)

            # Retrieve IoU, add to list
            self.iou_list.append(self.full_IoU(pred, pxd))     

            for j in range(self.num_classes+1):
                self.class_iou_list.append(self.classwise_IoU(pred[:,:,j], pxd[:,:,j]))
                  
        full_iou=np.mean(self.iou_list)
        return full_iou




    def full_DSC(self, hardmax, by):
        
        #(2tp/(2tp+fp+fn))
        
        intersection=np.sum(hardmax*by)
        twice_union=np.sum(hardmax)+np.sum(by)
        
        return (2*intersection)/(twice_union+0.0000001)
        
    def classwise_DSC(self, hardmax, by):
        
        #(2tp/(2tp+fp+fn))
        
        intersection=np.sum(hardmax*by)
        twice_union=np.sum(hardmax)+np.sum(by)
        
        return (2*intersection)/(twice_union+0.0000001)

    def full_IoU(self, hardmax, by):
        #both inputs are [1, h, w, num_classes]
        
        intersection=np.sum(hardmax*by)

        union=(hardmax+by)
        union[union>1]=1
        union=np.sum(union)

        IOU=intersection/(union+0.000001)
        
        return IOU

    def classwise_IoU(self, hardmax, by):
        # both inputs are [h, w]
        
        intersection=np.sum(hardmax*by)
        
        union=(hardmax+by)
        union[union>1]=1
        union=np.sum(union)

        IOU=intersection/union

        return IOU




    def viz(self, suffix, mode='Pred'):
        self.viz_list=[];
        if mode=='Pred':
            for i in range(len(self.pred_list)):
                viz=self.pred_list[i].viz()
                self.viz_list.append(viz)

        if mode=='Ground':
            for i in range(len(self.ground_list)):
                viz=self.ground_list[i].viz()
                self.viz_list.append(viz)
                
        self.export_workflow(suffix, mode=mode)
        return




    def export_workflow(self, suffix, mode='Pred'):
        for i in range(len(self.viz_list)):
            ID=str(i)
            while len(ID)<6:
                ID='0'+ID
            imageio.imwrite('./Results/Plots_' + mode + '/' + ID + '_' + suffix + '.png', self.viz_list[i].astype(np.uint8))
        return






