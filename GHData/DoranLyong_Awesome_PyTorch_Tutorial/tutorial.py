import torch 

from libs.data.fashion_mnist import * 
from libs.utils.figure import plt 
from libs import models 
from libs.models.d2l_alexnet import d2l_AlexNet
from libs.optim import optimizer, lr_scheduler
from libs import engine





"""
1. DataLoader 
"""
train_iter, test_iter = load_data_fashion_mnist(batch_size=128, resize=224)

print("num of train: " , len(train_iter.dataset))
print("num of test: ", len(test_iter.dataset))


imgs = [] 
lbls = [] 

for idx, data in enumerate(test_iter):
    if(idx>=0 and idx<10):
        imgs.append(data[0])
        lbls.append(data[1])

    if (idx>=10):
        break 

print("batch_size: ", len(lbls[0]))
#show_fashion_mnist(imgs[0], get_fashion_mnist_labels(lbls[0]))
num_classes = 10



"""
2. Build_model + optimizer + lr_scheduler 
"""

#_Start: backbone_check 
net = d2l_AlexNet()


X = torch.randn(size=(1,1,224,224))

for layer in net.model:
    X=layer(X)
    print(layer.__class__.__name__,'Output shape:\t',X.shape)

print(X)
print(X.max())
#_End: backbone_check 



model = models.build_model(
        model_name='d2l_alexnet',
        num_classes=10,
        loss='softmax',
        pretrained=True,
        )



optimizer = optimizer.build_optimizer(
            model=model,
            optim='sgd',
            lr=0.01,
            weight_decay=0,
            momentum=0,
            )


scheduler = lr_scheduler.build_lr_scheduler(
            optimizer=optimizer,
            lr_scheduler='single_step',
            stepsize=20
            )




"""
3. Build_engine : 
"""

engine = engine.ImageNLLEngine(
        datamanager=train_iter, 
        model=model, 
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True,
        )



"""
4. Run training & Testing 
"""
engine.run(
    save_dir='log', 
    max_epoch=5, 
    eval_freq=10, 
    print_freq=10, 
    test_only=False,
    )

"""
5. Testing
"""


