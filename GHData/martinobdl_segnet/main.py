import sys
sys.path.append("./model")
sys.path.append("./utils")
from deeplabv3 import DeepLabV3
import torch
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import cv2
import glob
from tqdm.autonotebook import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainId_to_id = {
    0: 7,
    1: 8,
    2: 11,
    3: 12,
    4: 13,
    5: 17,
    6: 19,
    7: 20,
    8: 21,
    9: 22,
    10: 23,
    11: 24,
    12: 25,
    13: 26,
    14: 27,
    15: 28,
    16: 31,
    17: 32,
    18: 33,
    19: 0
}

trainId_to_id_map_func = np.vectorize(trainId_to_id.get)

network = DeepLabV3("eval_val_for_metrics", project_dir=".").to(device)
if torch.cuda.is_available():
    network.load_state_dict(torch.load("./pretrained_models/model_13_2_2_2_epoch_580.pth"))
else:
    network.load_state_dict(torch.load("./pretrained_models/model_13_2_2_2_epoch_580.pth",
                                       map_location=torch.device('cpu')))


class Dataset(torch.utils.data.Dataset):
    def __init__(self):

        self.img_h = 512
        self.img_w = 1024

        self.examples = []

        file_names = glob.glob('../data/images_test/*jpg')
        for idx in range(len(file_names)):
            file_name = '../data/images_test/frame'+str(idx)+'.jpg'
            example = {}
            example["img_path"] = file_name
            example["img_id"] = idx
            self.examples.append(example)

        self.num_examples = len(file_names)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = example["img_path"]
        img = cv2.imread(img_path, -1)[196:-196, 232:-232]  # (shape: (1024, 2048, 3))
        # resize img without interpolation (want the image to still match
        # label_img, which we resize below):
        img = cv2.resize(img, (self.img_w, self.img_h),
                         interpolation=cv2.INTER_NEAREST)  # (shape: (512, 1024, 3))

        # resize label_img without interpolation (want the resulting image to
        # still only contain pixel values corresponding to an object class):

        # # # # # # # # debug visualization START
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("test", label_img)
        # cv2.waitKey(0)
        # # # # # # # # debug visualization END

        # normalize the img (with the mean and std for the pretrained ResNet):
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225])  # (shape: (512, 1024, 3))
        img = np.transpose(img, (2, 0, 1))  # (shape: (3, 512, 1024))
        img = img.astype(np.float32)

        # convert numpy -> torch:
        img = torch.from_numpy(img)  # (shape: (3, 512, 1024))

        return (img, img_id)

    def __len__(self):
        return self.num_examples


if __name__ == "__main__":
    save_folder = "../data/images_seg_test"
    batch_size = 2
    dataset = Dataset()
    num_batches = int(len(dataset)/batch_size)
    print("num_val_batches:", num_batches)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=1)
    network.eval()
    for (imgs, img_ids) in tqdm(data_loader):
        with torch.no_grad():
            imgs = Variable(imgs).to(device)  # (shape: (batch_size, 3, img_h, img_w))

            outputs = network(imgs)  # (shape: (batch_size, num_classes, img_h, img_w))

            ########################################################################
            # save data for visualization:
            ########################################################################

            outputs = outputs.data.cpu().numpy()  # (shape: (batch_size, num_classes, 1024, 2048))
            pred_label_imgs = np.argmax(outputs, axis=1)  # (shape: (batch_size, 1024, 2048))
            pred_label_imgs = pred_label_imgs.astype(np.uint8)

            for i in range(pred_label_imgs.shape[0]):
                pred_label_img = pred_label_imgs[i]  # (shape: (1024, 2048))
                img_id = img_ids[i]

                # convert pred_label_img from trainId to id pixel values:
                pred_label_img = trainId_to_id_map_func(pred_label_img)  # (shape: (1024, 2048))
                pred_label_img = pred_label_img.astype(np.uint8)
                name_out = save_folder + "/segm" + str(img_id.item()) + ".jpg"
                cv2.imwrite(name_out, pred_label_img)
