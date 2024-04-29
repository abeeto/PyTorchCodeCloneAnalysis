'''
Model Evaluation script. The evaluation strategy here is to show the prediction class of an image as an input image path is provided. Therefore, there is no need to use the DataLoader class to load the data. However, if you wish you evaluate in batches, use the LoadDataset class from load_data.py and DataLoader class to load the images. Note that the Evaluation script does not depend on any training parameters from train_cfg.
'''

import torch
import model
import eval_cfg as e_cfg
from image_transforms import ToTensor
from utils import read_image, evaluate_class


def main():
    '''
    Evaluation script.
    '''
    #Check if a trained model is present.
    assert e_cfg.TRAINED_FEAT_MODEL_PRESENCE, "There is no trained feature model present for evaluation! If a model is already placed in the appropriate folder, please check the name of the model file."
    assert e_cfg.TRAINED_CLASSIFIER_MODEL_PRESENCE, "There is no trained classifier model present for evaluation! If a model is already placed in the appropriate folder, please check the name of the model file."


    model_instance = model.Model(model_download_path=e_cfg.FEAT_MODEL_PATH, new_model_name=e_cfg.FEAT_MODEL_NAME, input_feature_size=e_cfg.FEATURE_INPUT_SIZE,
                                num_class=e_cfg.NUM_CLASSES, pretrained_download=False, freeze_feature_layer=True)

    vgg_model = model_instance()

    vgg_model = vgg_model.to(e_cfg.DEVICE)

    print("--- Model Architecture ---")
    print(vgg_model)

    #loads the model if a saved model.
    classifier_params = torch.load(e_cfg.CLASSIFIER_MODEL_PATH+e_cfg.CLASSIFIER_MODEL_NAME) #get
    vgg_model.load_state_dict(classifier_params)

    vgg_model.eval() #change the model to eval mode after loading the parameters. IMPORTANT STEP!

    print("Model classifier parameters are loaded from the saved file!")

    in_img = input("Please input the path of the image you wish to be evaluated: ")

    loaded_image = read_image(image_path=in_img, resized_image_size=e_cfg.RESIZED_IMAGE_SIZE) #load the image using cv2.

    tensor_image = ToTensor(mode='eval')({'image':loaded_image})['image'] #convert the loaded numpy image to Tensor using eval mode and extract only the image from the dict.

    #adds an extra dimension to emulate the batch size of 1 in the front and move the tensor to GPU if available.
    tensor_image = tensor_image.view(1, tensor_image.size()[0], tensor_image.size()[1], tensor_image.size()[2]).to(e_cfg.DEVICE)

    prediction_tensor = vgg_model(tensor_image) #output from the network.

    predicted_class = evaluate_class(net_output=prediction_tensor, classes_list=e_cfg.CLASSES) #get the predicted class.

    print(predicted_class)


if __name__ == '__main__':
    main()
