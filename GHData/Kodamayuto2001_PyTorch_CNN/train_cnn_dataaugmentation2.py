import random 
import shutil
import os 

class TestDataset:
    def __init__(self,root_train_dir="../SimpleNeuralNetwork_30_Classes/train/",root_dataset_dir="../30_dataset/dataset/"):
        test_sample_path = [
            [
                [], #   名前格納用
                []  #   path格納用
            ] for i in os.listdir(root_dataset_dir)
        ]

        self.result_random_sampling_path = [
            [
                [], #   名前格納用
                []  #   ランダムサンプリングパス
            ] for i in os.listdir(root_dataset_dir)
        ]

        for i,name in enumerate(os.listdir(root_dataset_dir)):
            for f in os.listdir(root_dataset_dir+name+"/"):
                path = root_dataset_dir + name + "/" + f
                test_sample_path[i][0]  =   name 
                test_sample_path[i][1].append(path)

        # self.printTestSamplePath(test_sample_path)

        name_list   =   self.nameListSet(root_train_dir)

        self.randomSampling(
            path_list=test_sample_path,
            name_list=name_list
        )

        # self.printResultRandomSampling()
        
    def getResult(self):
        return self.result_random_sampling_path

    def printTestSamplePath(self,test_list):
        for name in test_list:
            # print(name[0])
            if name[0] == "ando":
                print("ando")
            # for f in name[1]:
            #     print(f)

    def printResultRandomSampling(self):
        for x in self.result_random_sampling_path:
            for f in x[1]:
                print(f+"     "+x[0]+"     "+str(len(x[1])))
            

    def randomSampling(self,path_list,name_list,data_len=100):
        for i,p in enumerate(path_list):
            cnt = 0
            while True:
                if cnt == int(len(name_list)):
                    break
                if p[0] == name_list[cnt]:
                    #   名前格納
                    self.result_random_sampling_path[cnt][0]  =   name_list[cnt]
                    #   新しいリストに格納
                    self.result_random_sampling_path[cnt][1]  =   random.sample(p[1],data_len)
            
                #   リスト格納用番地を更新
                cnt += 1

        # for y in self.result_random_sampling_path:
        #     # print(y)
        #     for path in y[1]:
        #         print(path)
        #     # print(len(y[1]))

        self.result_random_sampling_path = [x for x in self.result_random_sampling_path if x != [[],[]]]

    def nameListSet(self,root_train_dir):
        return os.listdir(root_train_dir)


class NewTestDataset:
    def __init__(self,root_original_dataset="../30_dataset/original_X_5_24_dataset/",custom_train_20sheets_5places_24classes="../30_dataset/custom_train_20_5_24_dataset/"):
        dataset_path = [
            [
                [], #   名前格納用
                [], #   path格納用
                []  #   ファイル名格納用
            ] for i in os.listdir(root_original_dataset)
        ]
        train_path = [
            [
                [], #   名前格納用
                [], #   path格納用
                []  #   ファイル名格納用
            ] for i in os.listdir(custom_train_20sheets_5places_24classes)
        ]
        self.result_random_sampling_path = [
            [
                [], #   名前格納用
                [], #   path格納用
                []  #   ファイル名格納用
            ] for i in os.listdir(root_original_dataset)
        ]
        for i,name in enumerate(os.listdir(root_original_dataset)):
            for f in os.listdir(root_original_dataset+name+"/"):
                path = root_original_dataset + name + "/" + f
                dataset_path[i][0]  =   name 
                dataset_path[i][1].append(path)
                dataset_path[i][2].append(f)
        for i,name in enumerate(os.listdir(custom_train_20sheets_5places_24classes)):
            for f in os.listdir(custom_train_20sheets_5places_24classes+name+"/"):
                path = custom_train_20sheets_5places_24classes+name+"/"+f
                train_path[i][0] =   name
                train_path[i][1].append(path)
                train_path[i][2].append(f)

        self.removeTrainData(dataset=dataset_path,train_data=train_path)

    def removeTrainData(self,dataset,train_data):
        # print(dataset)
        # for d in dataset:
        #     print(d[0])
        #     # for p in d[1]:
        #     #     print(p)
        #     # for f in d[2]:
        #     #     print(f)
        # print(dataset[0][0])
        # print(dataset[1][0])
        cnt = 0
        while True:
            if cnt == len(dataset):
                break
            if dataset[cnt][0] == train_data[cnt][0]:
                f_cnt = 0
                while True:
                    if f_cnt == len(dataset[cnt][2]):
                        break
                    for i,t in enumerate(train_data[cnt][2]):
                        if t == dataset[cnt][2][f_cnt]:
                            #   datasetとtrain_dataが同じである
                            # print(dataset[cnt][1][f_cnt]+"     "+train_data[cnt][1][i])
                            # print(dataset[cnt][1][f_cnt]+"     "+t)
                            #   ファイル削除する
                            os.remove(dataset[cnt][1][f_cnt])
                            pass 
                        else:
                            # print(dataset[cnt][1][f_cnt])
                            pass 
                    f_cnt += 1
            cnt += 1
        # for cnt,d in enumerate(dataset):
        #     self.result_random_sampling_path[cnt][0].append(d[0])
        #     #   もしディレクトリの名前が同じなら
        #     if dataset[cnt][0] == train_data[cnt][0]:
        #         #   もしファイルの名前が同じなら
        #         # for f in dataset[cnt][1]:
        #         #     print(f)
        #         print(len(dataset[cnt][1]))
        #         # for f in train_data[cnt][1]:
        #         #     print(f)
        # print(self.result_random_sampling_path)
        # for x in self.result_random_sampling_path:
        #     print(x)
        pass 
            

class DataTo30Classes:
    NAME    =   [
        "ando",
        "uemura",
        "enomaru",
        "ooshima",
        "mizuki",
        "okamura",
        "kataoka",
        "kodama",
        "shinohara",
        "suetomo",
        "takemoto",
        "tamejima",
        "nagao",
        "hamada",
        "masuda",
        "matuzaki",
        "miyatake",
        "soushi",
        "ryuuga",
        "yamaji",
        "yamashita",
        "wada",
        "watanabe",
        "teppei",
        "kawano",
        "higashi",
        "tutiyama",
        "toriyabe",
        "matui",
        "ishino",
    ]
    CLASSES =   [
        [] for i in NAME
    ]

    def __init__(self,rootDir,saveDir):
        self.__classification(rootDir=rootDir)
        self.__save(rootDir,saveDir)


    def __classification(self,rootDir):
        name_list   =   os.listdir(rootDir)
        
        for i,classes in enumerate(self.NAME):
            tmp_list    =   []
            for n in name_list:
                try:
                    if classes[0]+classes[1]+classes[2]+classes[3]+classes[4]+classes[5] == n[0]+n[1]+n[2]+n[3]+n[4]+n[5]:
                        tmp_list.append(n)
                except IndexError:
                    if classes[0]+classes[1]+classes[2]+classes[3] == n[0]+n[1]+n[2]+n[3]:
                        tmp_list.append(n)
            self.CLASSES[i] =   tmp_list
        
        # for c in self.CLASSES:
        #     print(c)

    def __save(self,rootDir,saveDir):
        #####   -----   親ディレクトリ作成    -----   #####
        try:
            os.makedirs(saveDir)
        except FileExistsError:
            pass 
        for i,c in enumerate(self.NAME):
            #####   -----   子ディレクトリ作成    -----   #####
            try:
                os.makedirs(saveDir+c+"/")
            except FileExistsError:
                pass 
            # print(self.CLASSES[i])
            # print(len(self.CLASSES[i]))
            # print(saveDir+c+"/")

            cnt     =   0

            for subDirName in self.CLASSES[i]:
                # print(subDirName)
                # print(rootDir+subDirName+"/")
                for f in os.listdir(rootDir+subDirName+"/"):
                    path = rootDir+subDirName+"/"+f
                    # print(path)
                    # print(saveDir+c+"/"+f)
                    # print(cnt)
                    shutil.copyfile(path,saveDir+c+"/"+str(cnt)+".jpg")
                    cnt += 1
                pass 
 

class SaveTestDataset:
    def __init__(self,test_dataset_path_list,saveDir):
        try:
            os.makedirs(saveDir)
        except FileExistsError:
            pass 

        for name in test_dataset_path_list:
            try:
                os.makedirs(saveDir+name[0]+"/")
            except FileExistsError:
                pass 
            cnt = 0
            for f in name[1]:
                print(f)
                shutil.copyfile(f,saveDir+name[0]+"/"+str(cnt)+".jpg")
                cnt += 1

import train_cnn_dataaugmentation as T        

if __name__ == "__main__":
    # test    =   TestDataset()
    # print(test.getResult())
    # new_test    =   NewTestDataset()
    # d   =   DataTo30Classes(rootDir="../30_dataset/removedTrainData_X_5_24_dataset/",saveDir="../30_dataset/test_dataset_from_removedTrainData/")
    # test2   =   TestDataset(
    #     root_dataset_dir="../30_dataset/test_dataset_from_removedTrainData2/"
    # )
    # saveTestDataset =   SaveTestDataset(
    #     test_dataset_path_list=test2.getResult(),
    #     saveDir="../30_dataset/test_20_100sheets_dataset/"
    # )
    train_cnn_20_classes_data_augmentation = T.AI_30Classes(
        root_train_dir="../30_dataset/train_20classes_DA_10000/",
        root_test_dir="../30_dataset/test_20classes_DA_100/",
        PT_NAME="model_cnn_20classes_dataaugmentation.pt",
        LOSS_PNG="loss_cnn_20classes_dataaugmentation.png",
        ACC_PNG="acc_cnn_20_classes_dataaugmentation.png"
    )
    pass 