from dataset_loader import ImagesLoader
import torch
from mnist_model import Model
testing_paths = []
testing_paths.append("C:/Users/Samuel/PycharmProjects/pythonProject/MNIST/MNIST - JPG - testing/0/")
testing_paths.append("C:/Users/Samuel/PycharmProjects/pythonProject/MNIST/MNIST - JPG - testing/1/")
testing_paths.append("C:/Users/Samuel/PycharmProjects/pythonProject/MNIST/MNIST - JPG - testing/2/")
testing_paths.append("C:/Users/Samuel/PycharmProjects/pythonProject/MNIST/MNIST - JPG - testing/3/")
testing_paths.append("C:/Users/Samuel/PycharmProjects/pythonProject/MNIST/MNIST - JPG - testing/4/")
testing_paths.append("C:/Users/Samuel/PycharmProjects/pythonProject/MNIST/MNIST - JPG - testing/5/")
testing_paths.append("C:/Users/Samuel/PycharmProjects/pythonProject/MNIST/MNIST - JPG - testing/6/")
testing_paths.append("C:/Users/Samuel/PycharmProjects/pythonProject/MNIST/MNIST - JPG - testing/7/")
testing_paths.append("C:/Users/Samuel/PycharmProjects/pythonProject/MNIST/MNIST - JPG - testing/8/")
testing_paths.append("C:/Users/Samuel/PycharmProjects/pythonProject/MNIST/MNIST - JPG - testing/9/")


if __name__ == '__main__':
    loader = ImagesLoader(128)
    dataset = loader.get_dataset(testing_paths, training=False)
    model = Model()
    PATH = "./MNIST-MY.pth"
    model.load_state_dict(torch.load(PATH))

    def test_model():
        with torch.no_grad():

            good_predictions = 0
            for images, labels in (zip(*dataset)):

                y = model(images)
                y = torch.argmax(y, dim=1)

                difference = torch.eq(y, labels)

                for i in difference:
                    if i == True:
                        good_predictions+=1

            print("Accuracy is - ",(good_predictions)/100,"% and model missed", 10000- good_predictions,"of 10 000 images")

    test_model()
