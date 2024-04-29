import os
import numpy as np
import time










class Data_Paths():

    def __init__(self, pre_cached=True, train_mode=True):

        self.train_mode=train_mode;
        self.pre_cached=pre_cached
        self.sup_pools=[];
        



        
        # If training with real data only
        if train_mode==True: 
            self.sup_pools.append('/home/will/Desktop/OLD/Multiclass/Train/Sagittal/') 
            for i in range(28):
                self.sup_pools.append('/media/will/TOSHIBA EXT/DATA/Flinders_Seg/Training/' + str(i+1) + '/Sag/')
                self.sup_pools.append('/media/will/TOSHIBA EXT/DATA/Flinders_Seg/Training/' + str(i+1) + '/Cor/')
            #for jj in range(10):
            self.num_pools=len(self.sup_pools)
                    
        
        
        
        # 
        self.inf_out_pools=[];
        self.subj_num=[];
        self.scan_num=[];
        self.top_path2='/home/will/Desktop/Flinders_Seg/V3/Results/Inference/';
        if train_mode==False:
            top_path='/media/will/TOSHIBA EXT/DATA/Flinders_Seg/Inference/';
            subjs=os.listdir(top_path)
            for kk in range(len(subjs)):
                for jj in range(1,3):
                    self.sup_pools.append(top_path + subjs[kk] + '/' + str(jj) + '/');  
                    self.inf_out_pools.append(self.top_path2 + subjs[kk] + '/' + str(jj) + '/'); 
                    self.subj_num.append(str(subjs[kk]));
                    self.scan_num.append(str(jj));
                    #print(top_path + subjs[kk] + '/')
            self.num_pools=len(self.sup_pools)
        
        
        
        if self.pre_cached==True:
            self.import_paths()
        else:
            self.update_paths()
        
        
        return

    


    def update_paths(self):
        print('UPDATING PATHS')
        self.export_paths()
        self.import_paths()
        return




    def export_paths(self):

        if self.train_mode==True:
            suffix='train'
        else:
            suffix='val'

        print('EXPORTING SUPERVISED PATHS')
        # Get paths to images and export
        for i in range(len(self.sup_pools)):
            #print(i)
            t1=time.time()
            #self.path_list_sup.append()
            ID_list=os.listdir(self.sup_pools[i] + '/Imd/')
            #print(time.time()-t1)

            with open('./paths/' + str(i) + '_sup_' + str(suffix) + '.txt', 'w') as f:
                for j in range(len(ID_list)):
                    f.write("%s\n" % ID_list[j])
        return





    def import_paths(self):
        self.path_list_sup=[];
        self.path_list_uns=[];
        self.num_inst_sup=[];
        self.num_inst_uns=[];

        if self.train_mode==True:
            suffix='train'
        else:
            suffix='val'
        
        for i in range(len(self.sup_pools)):
            temp=[];
            file = open('./paths/' + str(i) + '_sup_' + str(suffix) + '.txt', mode='r')
            for i in file:
                temp.append(i[:-1])
            file.close()
            self.path_list_sup.append(temp)
            self.num_inst_sup.append(len(temp))

        return






    def sample_sup_ID(self, pool_num=None, ct=None):

        if pool_num==None and ct==None:
            if self.num_pools==1:
                rand_pool=0
            else:
                rand_pool=np.random.randint(low=0, high=self.num_pools)

            #rand_im=np.random.randint(low=int(len(self.path_list_sup[rand_pool])*0.35), high=int(len(self.path_list_sup[rand_pool])*0.65))
            rand_im=np.random.randint(low=1, high=int(len(self.path_list_sup[rand_pool])))
            im_path=self.path_list_sup[rand_pool][rand_im]
            # Extract ID
            ID=im_path.split('.')[0]
            data_root=self.sup_pools[rand_pool] 
                       
            if rand_pool==0:    
                lm_coeff=0;
            else:
                lm_coeff=1;                
            
        else:
            lm_coeff=[];
            data_root=self.sup_pools[pool_num] 
            ID=str(ct+1)
            while len(ID)<6:
                ID='0'+ID
            
            '''im_path=self.path_list_sup[rand_pool][rand_im]
            # Extract ID
            ID=im_path.split('.')[0]
            data_root=self.sup_pools[rand_pool]'''
            
        return data_root, ID, lm_coeff











'''
def unit_test():

    data_stuff=Data_Paths(local_mode=False)
    #data_stuff.export_paths()

    data_stuff.import_paths()

    print(data_stuff.sup_pools)
    print(data_stuff.uns_pools)
    print('')
    
    #print(data_stuff.path_list_sup)
    #print(data_stuff.path_list_uns)    

    print(data_stuff.sample_sup_ID())
    print(data_stuff.sample_uns_path())
    print('')

    print(data_stuff.num_inst_sup, data_stuff.num_inst_uns)

    return
'''



#unit_test()











 



