import cv2
import numpy as np
import glob
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms.functional as TF
from PIL import Image

from nets.GoogLeNet import GoogLeNet
from nets.VGGNet import VGGNet
from nets.ResNet import ResNet
from nets.DenseNet import DenseNet
import utils


def is_it_empty_in_night(canny_image, threshold):
    n_white_pix = np.sum(canny_image == utils.MAX_BRIGHTNESS)
    if n_white_pix > threshold:  # there is almost no edges
        return False
    else:
        return True


class MyNet:

    def __init__(self, pretrained=False, dimensions=3, net_type="idk", batch_size=8, epoch=3, img_size=96):
        self.data = []
        self.dimensions = dimensions  # grayscale=1 / rgb=3
        self.type = net_type
        self.batch_size = batch_size
        self.epoch = epoch
        self.img_size = img_size
        self.pretrained = pretrained

        self.path = 'trained_nets/' + net_type + '_e' + str(epoch) + '_d' + str(dimensions) + '_s' + str(
            self.img_size) + '.pth'
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.7, hue=0),
        ])

    def train(self):
        data_dir = 'train_images'
        image_datasets = datasets.ImageFolder(data_dir, transform=self.transform)
        data_loader = torch.utils.data.DataLoader(
            image_datasets, batch_size=self.batch_size, shuffle=True, num_workers=4)
        images, labels = iter(data_loader).next()
        # classes = ('free', 'full')
        # print("\t First batch:", end=" ")
        # print(' '.join('%5s' % classes[labels[j]] for j in range(self.batch_size)))
        utils.imshow(torchvision.utils.make_grid(images))

        # net types
        if self.type == "GoogLeNet":
            if not self.pretrained:
                net = GoogLeNet(3).net
            else:
                net = models.googlenet(pretrained=True)
        elif self.type == "VGGNet":
            if not self.pretrained:
                net = VGGNet(3).net  # models.googlenet(pretrained=True)
            else:
                net = models.vggnet(pretrained=True)
        elif self.type == "ResNet":
            if not self.pretrained:
                net = ResNet(3).net  # models.googlenet(pretrained=True)
            else:
                net = models.resnet18(pretrained=True)
        elif self.type == "DenseNet":
            if not self.pretrained:
                net = DenseNet(3).net  # models.googlenet(pretrained=True)
            else:
                net = models.resnet18(pretrained=True)
        else:
            print("E: I dont know this type", self.type)
            return

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(" - Using", device)
        net.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        print(" - Training started:", self.type, "with", self.epoch, "epochs and", self.dimensions, "dimensions")

        for epoch in range(self.epoch):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(data_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()  # zero the parameter gradients
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 20 == 19:  # print every 2000 mini-batches
                    print('\t[%d, %5d/520] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 20), end="\r")
                    running_loss = 0.0

        print(' - Training finished\t\t')
        torch.save(net, self.path)
        print(' -', self.type, 'saved to', self.path)

    def test(self):
        actual_results = utils.get_true_results()  # ground truth

        predicted_results = []  # net results
        iii = 0  # iterator
        fp = 0
        fn = 0
        tp = 0
        tn = 0

        if not self.pretrained:
            net = torch.load(self.path)
            print(" -", self.type, "loaded from", self.path)
        else:
            print("-", self.type, "loaded from torch models")
            if self.type == "GoogLeNet":
                net = models.googlenet(pretrained=True)
            elif self.type == "VGGNet":
                net = models.vggnet(pretrained=True)
            elif self.type == "ResNet":
                net = models.resnet18(pretrained=True)
        net.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)

        test_images = [img for img in glob.glob("test_images/*.jpg")]
        test_images.sort()

        parking_lot_coordinates = utils.get_coordinates()

        print(" - Testing started")
        for img in test_images:
            one_park_image = cv2.imread(img)
            one_park_image_show = one_park_image.copy()

            for parking_spot_coordinates in parking_lot_coordinates:
                pts_float = utils.get_points_float(parking_spot_coordinates)
                pts_int = utils.get_points_int(parking_spot_coordinates)
                warped_image = utils.four_point_transform(
                    one_park_image, np.array(pts_float))
                res_image = cv2.resize(warped_image, (self.img_size, self.img_size))
                one__img = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(one__img)

                image_pytorch = self.transform(img_pil).to(device)
                image_pytorch = image_pytorch.unsqueeze(0)
                output_pytorch = net(image_pytorch)

                _, predicted = torch.max(output_pytorch, 1)
                spotted_car = predicted[0]
                predicted_results.append(spotted_car)

                if actual_results[iii] and spotted_car:
                    tp += 1
                    utils.draw_cross(one_park_image_show, pts_int)
                if actual_results[iii] and not spotted_car:
                    fn += 1
                    utils.draw_rect(one_park_image_show, pts_int, utils.COLOR_BLUE)
                if not actual_results[iii] and spotted_car:
                    fp += 1
                    utils.draw_cross(one_park_image_show, pts_int, utils.COLOR_BLUE)
                if not actual_results[iii] and not spotted_car:
                    tn += 1
                    utils.draw_rect(one_park_image_show, pts_int)

                iii += 1

            cv2.imshow('one_park_image', one_park_image_show)

            key = cv2.waitKey(0)
            if key == 27:  # exit on ESC
                break

        print(" - Test results for ", self.type, "with", self.epoch, "epochs and", self.dimensions, "dimensions")

        eval_result = utils.get_parking_evaluation(
            tp, tn, fp, fn, iii)
        utils.print_evaluation_header()
        utils.print_evaluation_result(eval_result)

    def canny_test(self, threshold):

        actual_results = utils.get_true_results()  # ground truth

        predicted_results = []  # net results
        iii = 0  # iterator
        fp = 0
        fn = 0
        tp = 0
        tn = 0

        if not self.pretrained:
            try:
                net = torch.load(self.path)
                print(" -", self.type, "loaded from", self.path)
            except FileNotFoundError:
                print("E: No", self.path, "is trained!")
                return
            except:
                print("E: Problem loading net!")
                return

        else:
            print(" -", self.type, "loaded from torch models")

            if self.type == "GoogLeNet":
                net = models.googlenet(pretrained=True)
            elif self.type == "VGGNet":
                net = models.vggnet(pretrained=True)
            elif self.type == "ResNet":
                net = models.resnet18(pretrained=True)
        net.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)

        test_images = [img for img in glob.glob("test_images/*.jpg")]
        test_images.sort()

        parking_lot_coordinates = utils.get_coordinates()

        print(" - Testing with Canny started")
        for img in test_images:
            one_park_image = cv2.imread(img)
            one_park_image_show = one_park_image.copy()

            for parking_spot_coordinates in parking_lot_coordinates:
                pts_float = utils.get_points_float(parking_spot_coordinates)
                pts_int = utils.get_points_int(parking_spot_coordinates)
                warped_image = utils.four_point_transform(one_park_image, np.array(pts_float))
                res_image = cv2.resize(warped_image, (self.img_size, self.img_size))

                blur_image = cv2.GaussianBlur(res_image, (5, 5), 0)
                canny_image = cv2.Canny(blur_image, 40, 100)
                by_canny = False

                if is_it_empty_in_night(canny_image, threshold):
                    spotted_car = 0
                    predicted_results.append(spotted_car)
                    # cv2.imshow('canny_image', canny_image)
                    by_canny = True

                else:
                    one__img = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(one__img)
                    image_pytorch = self.transform(img_pil).to(device)
                    image_pytorch = image_pytorch.unsqueeze(0)
                    output_pytorch = net(image_pytorch)

                    _, predicted = torch.max(output_pytorch, 1)
                    spotted_car = predicted[0]
                    predicted_results.append(spotted_car)

                if actual_results[iii] and spotted_car:
                    tp += 1
                    if by_canny:
                        utils.draw_dotted_cross(one_park_image_show, pts_int, utils.COLOR_GREEN)
                    else:
                        utils.draw_cross(one_park_image_show, pts_int)

                    # print("TP")
                if actual_results[iii] and not spotted_car:
                    fn += 1
                    if by_canny:
                        utils.draw_dotted_rect(one_park_image_show, pts_int, utils.COLOR_BLUE)
                    else:
                        utils.draw_rect(one_park_image_show, pts_int, utils.COLOR_BLUE)
                if not actual_results[iii] and spotted_car:
                    fp += 1
                    if by_canny:
                        utils.draw_dotted_cross(one_park_image_show, pts_int, utils.COLOR_BLUE)
                    else:
                        utils.draw_cross(one_park_image_show, pts_int, utils.COLOR_BLUE)
                if not actual_results[iii] and not spotted_car:
                    tn += 1
                    if by_canny:
                        utils.draw_dotted_rect(one_park_image_show, pts_int, utils.COLOR_GREEN)
                    else:
                        utils.draw_rect(one_park_image_show, pts_int)
                iii += 1

            cv2.imshow('one_park_image', one_park_image_show)

            key = cv2.waitKey(0)
            if key == 27:  # exit on ESC
                break

        print(" - Testing finished for", self.type, "with", self.epoch, "epochs and", self.dimensions, "dimensions")

        eval_result = utils.get_parking_evaluation(tp, tn, fp, fn, iii)
        utils.print_evaluation_header()
        utils.print_evaluation_result(eval_result)
