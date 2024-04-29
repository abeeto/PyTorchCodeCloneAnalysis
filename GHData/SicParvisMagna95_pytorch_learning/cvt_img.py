import torchvision.datasets as Datasets
import matplotlib.pyplot as plt
import os



if __name__ == "__main__":
    # vgg16 = models.vgg16(pretrained=True,num_class=10)
    mnist_train = Datasets.MNIST('/home/chauncy/data', train=True, download=False)
    print('train set: ', len(mnist_train))


    train_label = open('/home/chauncy/data/MNIST/img/train_label.txt','w')
    for i,(img,label) in enumerate(mnist_train):
        img_path = os.path.join('/home/chauncy/data/MNIST/img/train_img', str(i).zfill(5)+'.jpg')
        img.save(img_path)
        train_label.write(img_path+'\t'+str(label)+'\n')

    train_label.close()


