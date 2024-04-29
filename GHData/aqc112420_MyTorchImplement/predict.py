import os
from args import MyArgumentParser
from Utils import data_classes, decode_labels
from Transforms import *
import glob
from scipy.misc import imread
import matplotlib.pyplot as plt
from models.Enet.Enet import ENet

args = MyArgumentParser()

transforms = Compose(
        [Resize(args.height, args.width),
         Normalize(mean=args.mean, std=args.std),
         ToTensor()])


def predict(img_paths, model, train_model_path):
    model.to(args.device)
    checkpoint = torch.load(train_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    miou = checkpoint['miou']
    print(epoch)
    print(miou)
    for path in img_paths[:2]:
        img = imread(path)
        lbl = imread(path.replace("test", "testannot"))
        img, lbl = transforms(img, lbl)
        img = img.unsqueeze(0)
        img = img.to(args.device)
        # Make predictions!
        model.eval()
        with torch.no_grad():
            predictions = model(img)

        # Predictions is one-hot encoded with "num_classes" channels.
        # Convert it to a single int using the indices where the maximum (1) occurs
        _, predictions = torch.max(predictions.data, 1)
        print(predictions.data.cpu().numpy().shape)
        predict = predictions.data.cpu().numpy()[0]
        plt.imshow(decode_labels(predict, cmap=cmap))
        plt.show()
        plt.imsave(args.images_dir+path.split('\\')[-1].replace('png', 'jpg'), decode_labels(predict, cmap=cmap))



if __name__ == '__main__':
    img_paths = glob.glob(r'E:\anqc\datasets\CamVid480x360\test\*.png')
    model = ENet(args.num_classes)
    _, cmap = data_classes('camvid')
    if not os.path.exists(args.images_dir):
        os.mkdir(args.images_dir)
    predict(img_paths, model, "./model/ENet.pth")




# def predict(model, images, class_encoding):
#     images = images.to(args.device)
#     # Make predictions!
#     model.eval()
#     with torch.no_grad():
#         predictions = model(images)
#
#     # Predictions is one-hot encoded with "num_classes" channels.
#     # Convert it to a single int using the indices where the maximum (1) occurs
#     _, predictions = torch.max(predictions.data, 1)
#
#     label_to_rgb = transforms.Compose([
#         ext_transforms.LongTensorToRGBPIL(class_encoding),
#         transforms.ToTensor()
#     ])
#     color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
#     utils.imshow_batch(images.data.cpu(), color_predictions)
