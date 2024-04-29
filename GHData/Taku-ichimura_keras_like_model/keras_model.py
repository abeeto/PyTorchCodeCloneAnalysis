import os
import time
import json
import torch
class keras_model():
    """
    keras風のトレーニング+出力にするクラス
    GPU計算+1GPUが前提
    
    Attributes
    ----------
    model : net
        pytorchで作ったモデル
    epoch : int
            epoch数
    dataloader : DataLoader
        訓練データ
    criterion : torch.nn
        lossの種類
    optimizer : optim
        pytorchのoptimizerクラス
    scheduler : scheduler
        putorchのschedulerクラス
    validation : DataLoader
        validationデータ
    accuracy : function
        accuracyを返す関数。デフォルトはクラス分類を想定(=出力がスカラ)。
        引数と出力はdefault_accuracyと同じである必要がある
    step_num : int
        step数
    epoch_result : dict
        epochごとの結果を格納する辞書
    best_result : dict
        一番良い結果の値を格納する辞書
    """
    def __init__(self,model=None):
        """
        Parameters
        ----------
        model : net
            pytorchのモデル
        """
        self.model = model
        self.epoch = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.dataloader = None
        self.accuracy = None
        self.validation = None 
        self.step_num = None
        self.epoch_result = {"loss":[],"acc":[],"val_loss":[],"val_acc":[]}
        self.best_result = {"loss":2**63-1,"acc":0,"val_loss":2**63-1,"val_acc":0}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model is None:
            raise ValueError("model is None!")
    
    def train_step(self,images,labels,val=False):
        """
        １step分の学習をする

        Parameters
        ----------
        images : Tensor
            (batch_size,C,H,W)のTensor
        labels : Tensor
            正解ラベルのTensor
        val : bool ,default False
            Trueのとき評価モード
        Returns
        -------
        pred : Tensor
            推論された値のTensor
        loss : float
            lossの値
        """
        optimizer = self.optimizer
        model = self.model
        criterion = self.criterion
        scheduler = self.scheduler
        device = self.device
        
        if val:
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            loss = criterion(out, labels)
            return out.max(1)[1],loss.item()
        
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        return out.max(1)[1],loss.item()
    
    
    def fit(self,epoch=None,dataloader=None,criterion=None,optimizer=None,scheduler=None,validation=None,accuracy=None,save_type="no_auto_save",save_path="./"):
        """
        モデルの学習を実行
        Parameters
        ----------
        epoch : int
            epoch数
        dataloader : DataLoader
            訓練データ
        criterion : criterion
            lossの種類
        optimizer : optimizer
            pytorchのoptimizerクラス
        scheduler : scheduler or None, default None
            putorchのschedulerクラス、指定は任意
        validation : DataLoader or None, default None
            validationデータ、指定は任意
        accuracy : function or None, default None
            accuracyを返す関数、指定は任意。デフォルトはクラス分類を想定(=出力がスカラ)。
            引数と出力はdefault_accuracyと同じである必要がある
        save_type : string or list, default "no_auto_save"
            モデルを保存する規則を指定
            loss,acc,val_loss,val_accを指定でき、指定した値が最高のときbest_[save_type].pthに保存
            複数指定する場合はlistに入れる
        save_path : string, default "./"
            重みをセーブするときのpath
        """
        self.epoch = epoch
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accuracy = accuracy
        self.validation = validation 
        self.step_num = len(dataloader)
        
        if epoch is None:
            raise ValueError("epoch is not None")
        
        if dataloader is None:
            raise ValueError("dataloader is not None")
        
        if optimizer is None:
            raise ValueError("optimizer is not None")

        for e in range(epoch):
            print("Epoch {}/{}".format(e+1,epoch))
            step_t=0
            step_acc=0
            step_loss=0
            n_sample=0
            val_acc=None
            val_loss=None
            self.model.train()
            for i,(batch_images,batch_labels) in enumerate(dataloader):
                start_step_t = time.time()
                preds,loss = self.train_step(batch_images,batch_labels)
                if accuracy is None:
                    acc = self._default_accuracy(preds,batch_labels)
                else:
                    acc = accuracy(preds,batch_labels)
                end_step_t = time.time()
                step_t+=end_step_t - start_step_t
                step_acc+=acc
                step_loss+=loss
                n_sample+=len(batch_labels)
                acc = step_acc/n_sample
                loss = step_loss/(i+1)
                self._keras_like_output(step=i,acc=acc,loss=loss,t=step_t)
            self.epoch_result["loss"].append(loss)
            self.epoch_result["acc"].append(acc)
            
            if validation is not None:
                v_loss=0
                v_acc=0
                n_sample=0
                self.model.eval()
                with torch.no_grad():
                    for i,(batch_images,batch_labels) in enumerate(validation):
                        val_preds,val_loss = self.train_step(batch_images,batch_labels,True)
                        if accuracy is None:
                            val_acc = self._default_accuracy(val_preds,batch_labels)
                        else:
                            val_acc = accuracy(val_preds,batch_labels)
                        v_acc+=val_acc
                        v_loss+=val_loss
                        n_sample+=len(batch_labels)
                    val_acc = v_acc/n_sample
                    val_loss = v_loss/(i+1)
                    self._keras_like_output(step=i,acc=acc,loss=loss,val_loss=val_loss,val_acc=val_acc,t=step_t,val=True)
                self.epoch_result["val_loss"].append(val_loss)
                self.epoch_result["val_acc"].append(val_acc)
            is_update = self.save_weight(save_type,save_path,loss,acc,val_loss,val_acc)
            if is_update == True:
                print("")
                print("save weights")
            self.best_result["loss"] = min(self.best_result["loss"],loss)
            self.best_result["acc"] = max(self.best_result["acc"],acc)
            if validation is not None:
                self.best_result["val_loss"] = min(self.best_result["val_loss"],val_loss)
                self.best_result["val_acc"] = max(self.best_result["val_acc"],val_acc)
            print("")

    def save_weight(self,save_type,save_path,loss,acc,val_loss,val_acc):
        """
        lossや精度が良い重みを更新しながら保存する

        Parameters
        ----------
        save_type : string or list[string]
            保存タイプの指定、どの量が最良のときに更新して保存するかを指定(複数可)
        save_path : string
            保存先のフォルダのパス
        loss : float
            あるepochのloss
        acc : float
            あるepochのaccuracy
        val_loss : float
            あるepochのvalidation-loss
        val_acc : float
            あるepochのvalidation-accuracy

        Returns
        -------
        update_flag : bool
            重みをupdateしたかどうか(Trueのときしている）
        """
        update_flag = False
        if save_type=="no_auto_save":
            return None
        
        elif type(save_type) is list:
            for t in save_type:
                if not t in list(self.epoch_result.keys()):
                    raise ValueError("save_type is loss or acc or val_loss or val_acc")
                if t == "loss" and  self.best_result["loss"] > loss:
                    torch.save(self.model.state_dict(),os.path.join(save_path,"best_loss.pth"))
                    update_flag = True
                elif t == "acc" and  self.best_result["acc"] < acc:
                    torch.save(self.model.state_dict(),os.path.join(save_path,"best_acc.pth"))
                    update_flag = True
                
                if val_loss is not None:
                    if t == "val_loss" and  self.best_result["val_loss"] > val_loss:
                        torch.save(self.model.state_dict(),os.path.join(save_path,"best_val_loss.pth"))
                        update_flag = True
                if val_acc is not None:
                    if t == "val_acc" and  self.best_result["val_acc"] < val_acc:
                        torch.save(self.model.state_dict(),os.path.join(save_path,"best_val_acc.pth"))
                        update_flag = True
        
        elif save_type in list(self.epoch_result.keys()):
            if not save_type in list(self.epoch_result.keys()):
                    raise ValueError("save_type is loss or acc or val_loss or val_acc")
            if save_type == "loss" and  self.best_result["loss"] > loss:
                torch.save(self.model.state_dict(),os.path.join(save_path,"best_loss.pth"))
                update_flag = True
            elif save_type == "loss" and  self.best_result["acc"] < acc:
                torch.save(self.model.state_dict(),os.path.join(save_path,"best_loss.pth"))
                update_flag = True
            
            if val_loss is not None:
                if save_type == "val_loss" and  self.best_result["val_loss"] > val_loss:
                    torch.save(self.model.state_dict(),os.path.join(save_path,"best_val_loss.pth"))
                    update_flag = True
            if val_acc is not None:
                if save_type == "val_acc" and  self.best_result["val_acc"] < val_acc:
                    torch.save(self.model.state_dict(),os.path.join(save_path,"best_val_acc.pth"))
                    update_flag = True
        else:
            raise ValueError("save_type is string or list[string]")

        return update_flag
            

    def _default_accuracy(self,preds,labels):
        """
        accuracyを計算

        Parameters
        ----------
        preds : Tensor
            推論結果のTensor
        labels : Tensor
            正解ラベルのTensor

        Returns
        -------
        acc : float
            精度
        """
        acc = (preds == labels.to(self.device)).sum()
        return acc.item()
        
    def _keras_like_output(self,step,acc,loss,t,val=False,val_loss=None,val_acc=None):
        """
        kerasのverboseぽい出力

        Parameters
        ----------
        step : int
            何step目か
        acc : float
            各stepまでにおける精度の平均
        loss : float
            各stepまでにおけるlossの平均
        val : bool or, default False
            validation結果の出力か否か
        val_loss : float or None, default None
            validationのlossの平均
        val_acc : float or None, default None
            validationの精度
        """
        if val:
            l = self.step_num
            epoch = self.epoch
            step_p = (step+1)/l
            bar = "=" *30
            result = " - loss:{:.4f} - acc:{:.4f} - val_loss:{:.4f} - val_acc:{:.4f}".format(loss,acc,val_loss,val_acc)
            print("\r {}/{}[{}]-{:.2f}s{}".format(l,l,bar,t,result), end="")
        else:
            l = self.step_num
            epoch = self.epoch
            step_p = (step+1)/l
            bar = "=" *int(step_p*30)+">" + " " *(30-int(step_p*30)+1)
            if step_p==1:
                bar = "=" *30
            result = " - loss:{:.4f} - acc:{:.4f}".format(loss,acc)
            print("\r {}/{}[{}]-{:.2f}s{}".format(step+1,l,bar,t,result), end="")
