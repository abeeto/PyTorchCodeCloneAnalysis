import torch
import torchvision
from torchvision import models, transforms, datasets

import cv2
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import os

def load_checkpoint(filepath):

    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        
    model = models.resnet34(pretrained=True)
    #model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def process_image(image, video=False):
    
    if video == True:
        pil_im = image
    else:
        # Process a PIL image for use in a PyTorch model

        # Converting image to PIL image using image file path
        pil_im = Image.open(f'{image}' + '.jpg')

    # Building image transform
    transform = transforms.Compose([transforms.Resize((244,244)),
                                    #transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]) 
    
    # Transforming image for use with network
    pil_tfd = transform(pil_im)
    
    # Converting to Numpy array 
    array_im_tfd = np.array(pil_tfd)
    
    return array_im_tfd

def predict(image_path, model, topk=5):
    # Implement the code to predict the class from an image file   
    
    # Loading model - using .cpu() for working with CPUs
    loaded_model = load_checkpoint(model).cpu()
    # Pre-processing image
    img = process_image(image_path)
    # Converting to torch tensor from Numpy array
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Adding dimension to image to comply with (B x C x W x H) input of model
    img_add_dim = img_tensor.unsqueeze_(0)

    # Setting model to evaluation mode and turning off gradients
    loaded_model.eval()
    with torch.no_grad():
        # Running image through network
        output = loaded_model.forward(img_add_dim)
        
    #conf, predicted = torch.max(output.data, 1)   
    probs_top = output.topk(topk)[0]
    predicted_top = output.topk(topk)[1]
    
    # Converting probabilities and outputs to lists
    conf = np.array(probs_top)[0]
    predicted = np.array(predicted_top)[0]
        
    #return probs_top_list, index_top_list
    return conf, predicted

def find_classes(dir):

    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
   
    return classes, class_to_idx

def plot_solution(cardir, model):
  # Testing predict function

  # Inputs are paths to saved model and test image
  model_path = 'model.pth'
  image_path = test_dir + cardir
  carname = cardir.split('/')[-2]

  conf2, predicted1 = predict(image_path, model_path, topk=5)
  # Converting classes to names
  names = []
  for i in range(5):
    names += [classes[predicted1[i]]]


  # Creating PIL image
  image = Image.open(image_path+'.jpg')

  # Plotting test image and predicted probabilites
  f, ax = plt.subplots(2,figsize = (6,10))

  ax[0].imshow(image)
  ax[0].set_title(carname)

  y_names = np.arange(len(names))
  ax[1].barh(y_names, conf2/conf2.sum(), color='darkblue')
  ax[1].set_yticks(y_names)
  ax[1].set_yticklabels(names)
  ax[1].invert_yaxis() 

  plt.show()


def draw_label(img, true_classes, predicted_classes, cfscs, font_size):
   
    predicted_classes = [true_classes[i] for i in predicted_classes]
    h,w,c = img.shape
    offset = int(h/1.5)
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thick = 1 
    color = (0, 255, 0)
    margin = 2
    bg_color = (0,0,0)

    for class_name, cfsc in zip(predicted_classes, cfscs):
        
        text = '{} Conf: {:.4f}'.format(class_name, cfsc)
        
        offset += int(h / len(predicted_classes)) - 50
        txt_size = cv2.getTextSize(text, font_face, font_size, thick)
        end_x = 1 + txt_size[0][0] + margin
        end_y = offset - txt_size[0][1] - margin

        cv2.rectangle(img, (1, offset), (end_x, end_y), bg_color, cv2.FILLED)
        cv2.putText(img, text, (1, offset), font_face, font_size, color, thick, cv2.LINE_AA)
        
        cv2.putText(img, text, (1, offset), font_face, font_size, color, thick, cv2.LINE_AA)

if __name__ == "__main__":

    model = load_checkpoint('model.pth').cpu()
    model.eval()

    data_dir = "car_data"
    test_dir = data_dir + '/test'

    classes, c_to_idx = find_classes(data_dir + '/train')
    
    
    cap = cv2.VideoCapture('NISSAN 240SX.mp4')

    while True:

        ret, frame = cap.read()
        
        if ret == True:

            # print(predict_video(model, frame))

            pil_image = Image.fromarray(frame)
            transform = transforms.Compose([transforms.Resize((244,244)),
                                    #transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])
            pil_tfd = transform(pil_image)
            array_im_tfd = np.array(pil_tfd)
            img_tensor = torch.from_numpy(array_im_tfd).type(torch.FloatTensor)
            img_add_dim = img_tensor.unsqueeze_(0)
            
            with torch.no_grad():
                output = model.forward(img_add_dim)

            topk = 5
            predicted_top = output.topk(topk)[1]
            predicted = np.array(predicted_top)[0]

            probs_top = output.topk(topk)[0]
            cfscs = probs_top/probs_top.sum()
            cfscs = list(cfscs.detach().cpu().numpy().flat)
            
            #change text font size here (recommended [0-1])
            font_size = 0.3

            draw_label(frame, classes, predicted, cfscs, font_size) 
            cv2.imshow('video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows() 
