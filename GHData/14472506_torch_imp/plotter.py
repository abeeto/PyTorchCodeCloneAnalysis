import json
import matplotlib.pyplot as plt

def load_json(json_dir):
    
    with open(json_dir) as f:
        d = json.load(f)
    
    return d

def seed_fix():
    
    t1 = load_json("output/Mask_RCNN_fix_test1/results.json")
    t2 = load_json("output/Mask_RCNN_fix_test2/results.json")
    t3 = load_json("output/Mask_RCNN_fix_test3/results.json")
    t4 = load_json("output/Mask_RCNN_fix_test4/results.json")
    t5 = load_json("output/Mask_RCNN_fix_test5/results.json")

    plt.plot(t1['train_epoch'], t1['train_loss'], label="test1_train")
    plt.plot(t2['train_epoch'], t2['train_loss'], label="test2_train")
    plt.plot(t3['train_epoch'], t3['train_loss'], label="test3_train")
    plt.plot(t4['train_epoch'], t4['train_loss'], label="test4_train")
    plt.plot(t5['train_epoch'], t5['train_loss'], label="test5_train")
    #plt.show()

    plt.plot(t1['val_epoch'], t1['val_loss'], label="test1_val")
    plt.plot(t2['val_epoch'], t2['val_loss'], label="test2_val")
    plt.plot(t3['val_epoch'], t3['val_loss'], label="test3_val")
    plt.plot(t4['val_epoch'], t4['val_loss'], label="test4_val")
    plt.plot(t5['val_epoch'], t5['val_loss'], label="test5_val")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Mask R-CNN - Resnet50 - lr:0.005")
    plt.show()


def loss_plotter():
    
    data = load_json('output/Mask_RCNN_dev_test/results.json')
    
    to_epoch = 156
    
    train_epochs = [x / to_epoch for x in data['train_epoch']]
    val_epoch = [x / to_epoch for x in data['val_epoch']]
    
    plt.plot(train_epochs, data['train_loss'], label="train loss")
    plt.plot(val_epoch, data['val_loss'], label="val loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Mask R-CNN - Resnet50 - lr:0.005")
    plt.show()

if __name__ == "__main__":
    
    loss_plotter()
    #seed_fix()