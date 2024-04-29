#from sre_parse import CATEGORIES
# from statistics import mode
import numpy as np
import torch 
import torchvision as tv
import time
import os
import argparse

# pda imports 
import dataloader as udl 
import classifiers as ucls
import visualizer as uv
import sampler as sampling
from pred_diff_analyser import PredDiffAnalyser


from config import *
import matplotlib.pyplot as plt


# ood model
import model.ood.react.model_loader as jml

# 
import model.ood.isomax.resnet_configloss as rncl
import model.ood.isomax.losses as losses
from model.ood.isomax.ood import entropic_score_normalized_inverted, max_probability_score_inverted

def infoResNet(show=False,mVersion='resnet50'):
    '''
    ResNet model info
    '''

    #model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    if mVersion == 'resnet50':
        model = tv.models.resnet50(pretrained=True)
    elif mVersion == 'resnet18':
        model = tv.models.resnet18(pretrained=True)
    else:
        assert False, 'invaled resnet version'

    # define preprocess and disply corpping functions
    preprocess = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    
    display = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor()])

    return {'model' : model, 'name' : mVersion, 'preprocess' : preprocess, 'display' : display, 'path' : '', 'ood' : None}

def infoMobileNetV2(show=False):
    '''
    mobelNetV2 model info
    '''
    model = tv.models.mobilenet_v2(pretrained=True)
    if show: 
        print('eval model: \n', model)

    # model.eval()

    # define preprocess and disply corpping functions
    preprocess = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    
    display = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor()])

    return {'model' : model, 'name' : "moblienet_v2",'preprocess' : preprocess, 'display' : display, 'path' : '', 'ood' : None}

def infoAlexNet(show=False):
    '''
    mobelNetV2 model info
    '''
    model = tv.models.alexnet(pretrained=True, progress=True)

    if show: 
        print('eval model: \n', model)

    # define preprocess and disply corpping functions
    preprocess = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    
    display = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor()])

    return {'model' : model, 'name' : "alexnet",'preprocess' : preprocess, 'display' : display, 'path' : '', 'ood' : None}

def infoVgg16(show=False):
    '''
    mobelNetV2 model info
    '''
    model = tv.models.vgg16(pretrained=True, progress=True)

    if show: 
        print('eval model: \n', model)

    # define preprocess and disply corpping functions
    preprocess = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    
    display = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor()])

    return {'model' : model, 'name' : "vgg16",'preprocess' : preprocess, 'display' : display, 'path' : '', 'ood' : None}


def infoConradRSICD(path, show=False, dset='04_RSICD'):
    '''
    RSIDCD model info RESNET50
    '''
    def get_normalize_transform(dataset_name):
        norm_dict = {
            "01_mnist_fashion": tv.transforms.Normalize((0.3201, 0.3182, 0.3629), (0.1804, 0.3569, 0.1131)),
            "02_cifar10":  tv.transforms.Normalize((0.4881, 0.4660, 0.3994), (0.2380, 0.2322, 0.2413)),
            "03_sen12ms":  tv.transforms.Normalize((0.1674, 0.1735, 0.2059), (0.1512, 0.1152, 0.1645)),
            "rsicd"     :  tv.transforms.Normalize((0.3897, 0.4027, 0.3715), (0.2050, 0.1920, 0.1934)),
            "05_xView2":  tv.transforms.Normalize((0.3292, 0.3408, 0.2582), (0.1682, 0.1408, 0.1296)),
            "06_So2SatLCZ42":  tv.transforms.Normalize((0.2380, 0.3153, 0.5004), (0.0798, 0.1843, 0.0666)),
        }
        return norm_dict[dataset_name]



    model_path = path + dset + '-resnet50-isomax.pth'

    loss_first_part = losses.IsoMaxLossFirstPart
    model = rncl.resnet50(loss_first_part=loss_first_part, num_classes=23,input_channels=3)#
    model.load_state_dict(torch.load(model_path))
    if show: 
        print('eval model: \n', model)


    # define preprocess and disply corpping functions
    preprocess = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        #tv.transforms.Normalize(mean=[0.3897, 0.4027, 0.3715], std=[0.2050, 0.1920, 0.1934]),])
        get_normalize_transform(dset)])
    
    display = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor()])

    # define ood post proc method
    if OOD: 
        # ood = entropic_score_normalized_inverted
        ood = max_probability_score_inverted
    else:
        ood = None

    return {'model' : model, 'name' : "rsicd_conrad",'preprocess' : preprocess, 'display' : display, 'path' : path, 'ood' : ood}


def test_prob(classifier, x):
    '''
    test predict function 
    '''
    top_prob, top_catid = classifier.getTopCat(x[0:1])
    
    cat_prob = classifier.predict(x) 

    print(np.max(cat_prob), np.min(cat_prob))


def get_explenation(x, x_im, path, classifier, sampler, top_catid):
        start_time = time.time()

        pda = PredDiffAnalyser(x[np.newaxis], classifier, sampler, num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE)
        pred_diff = pda.get_rel_vect(win_size=WIN_SIZE, overlap=OVERLAPPING)

        # for now
        sensMap = np.zeros(x_im.shape)

        print ("--- Total computation took {:.4f} minutes ---".format((time.time() - start_time)/60))

        np.savez(path, *pred_diff)
        uv.plot_results(x, x_im, sensMap, pred_diff[0], classifier.categories, top_catid, path, show=SHOW)


def view_explenation(x, x_im,classifier, path, top_catid, top_prob, y_pred_label):
        '''
        load results and view
        '''

        path = path + '.npz'
        print('Try to load: ', path)


        try: 
            npFile = np.load(path)
            # print('Arrays in file:', npFile.files)
            pred_diff = npFile['arr_0']
            # print('array: ', pred_diff.shape)
            # not use, simply set to zero for now
            sensMap = np.zeros(x_im.shape)

            # get min max values from show class
            p_class = pred_diff[:,top_catid]
            print('MIN: {}/MAX: {}'.format(np.min(p_class), np.max(p_class)))

            print(y_pred_label, top_prob)
            uv.plot_results(x, x_im, sensMap, pred_diff, classifier.categories, top_catid, show=True)
        except:
            print('not processed: ', y_pred_label, top_prob)

def convert_uint8(path, x_name):
    '''
    convert pda map to uint8 for evaluation
    '''

    path = path + '.npz'
    print('Try to load: ', path)

    try: 
        npFile = np.load(path)
    except:
        print('not processed: ' + path)
        return

    # clip_val = 1.5 # for classification without softmax
    clip_val = 0.1 # for regression model
    

    pred_diff = npFile['arr_0']
    # print('o', pred_diff.shape, pred_diff[0:4,0])
    # clip to 1.5, seems sufficent
    clipped_diff = np.clip(pred_diff, -clip_val, clip_val)
    # print('c', clipped_diff[0:4,0])

    scaled_diff = ((clipped_diff / clip_val) * 127) + 128
    # print('s', scaled_diff[0:4,0])

    uint_diff = scaled_diff.astype(np.uint8)

    path_results = RESULT_PATH + 'uint8/'
    if not os.path.exists(path_results):
        os.makedirs(path_results)  

    np_path = path_results + '{}'.format(x_name)
    np.save(np_path,uint_diff)

    # test show image
    # uint_diff = np.moveaxis(uint_diff, -1, 0)
    # cv2.imshow("window", uint_diff.reshape(CLASSES, 224,224)[0])
    # cv2.waitKey(0)



def experiment(model, visualize=False, convert=False):

    dataLoader = udl.DataLoader(path=IMAGE_PATH)

    test_size = TESTS
    show      = SHOW
    softmax = SOFTMAX
    X_test,X_test_im, X_filenames = dataLoader.get_data(s_idx=IMG_IDX, b_size=test_size)

    path_results = RESULT_PATH
    if not os.path.exists(path_results):
        os.makedirs(path_results)  

    classifier = ucls.Classifier(model, softmax=softmax, categroies=CATEGORIE_PATH)
    sampler = sampling.ConditionalSampler(win_size=WIN_SIZE, padding_size=PADDING_SIZE, image_dims=(224,224), netname=classifier.name)


    for test_idx in range(test_size):

        x_test = X_test[test_idx]
        x_test_im = X_test_im[test_idx]
        x_test_path = X_filenames[test_idx]

        top_prob, top_catid = classifier.getTopCat(x_test[np.newaxis])
        y_pred_label = classifier.categories[top_catid]
        print('ylable', y_pred_label,'idx %d prob %.3f' %(top_catid,top_prob))

        path = path_results + '{}_{}_winSize{}_stride{}_sm{}_{}.jpg'.format(X_filenames[test_idx],y_pred_label,WIN_SIZE, STRIDE,int(softmax),classifier.name)
        print('using path', path)
        # perfom 
        if visualize:
            view_explenation(x_test, x_test_im,classifier, path, top_catid, top_prob, y_pred_label)
        elif convert:
            convert_uint8(path, x_test_path)
        else: # pda
            get_explenation(x_test, x_test_im, path, classifier, sampler, top_catid)


        # start_time = time.time()

        # pda = PredDiffAnalyser(x_test[np.newaxis], classifier, sampler, num_samples=NUM_SAMPLES, batch_size=BATCH_SIZE)
        # pred_diff = pda.get_rel_vect(win_size=WIN_SIZE, overlap=OVERLAPPING)

        # # for now
        # sensMap = np.zeros(x_test_im.shape)

        # print ("--- Total computation took {:.4f} minutes ---".format((time.time() - start_time)/60))

        # uv.plot_results(x_test, x_test_im, sensMap, pred_diff[0], classifier.categories, top_catid, save_path, show=SHOW)
        # np.savez(save_path, *pred_diff)

        # print('result:', save_path)
    



# def visualize(model, im_idx=IMG_IDX, test_classes=5):
#     '''
#     model: model for classifcation to get top cat id
#     im_idx : get specific image form folder
#     test_classes: number of random classes to be shown
#     '''

#     test_size = TESTS
#     show      = SHOW
#     softmax = SOFTMAX

#     path_results = RESULT_PATH


#     dataLoader = udl.DataLoader(model=model, path=IMAGE_PATH)

#     X_test,X_test_im, X_filenames = dataLoader.get_data(s_idx=im_idx, b_size=test_size)

#     classifier = ucls.Classifier(model, softmax=softmax, categroies=CATEGORIE_PATH)


#     for test_idx in range(test_size):

#         x_test = X_test[test_idx]
#         x_test_im = X_test_im[test_idx]
#         x_test_path = X_filenames[test_idx]

#         top_probs, top_catids = classifier.getTopCats(x_test[np.newaxis],top=test_classes)
#         top_prob   = top_probs[0][0].item()
#         top_catid  = top_catids[0][0].item()
#         y_pred_label = classifier.categories[top_catid]



#         load_path = path_results + '{}_{}_winSize{}_stride{}_sm{}_{}.jpg.npz'.format(X_filenames[test_idx],y_pred_label,WIN_SIZE, STRIDE,int(softmax),classifier.name)

#         print('Try to load: ', load_path)

#         try: 
#             npFile = np.load(load_path)
#             # print('Arrays in file:', npFile.files)
#             pred_diff = npFile['arr_0']
#             # print('array: ', pred_diff.shape)
#             # not use, simply set to zero for now
#             sensMap = np.zeros(x_test_im.shape)

#             # get min max values from show class
#             p_class = pred_diff[:,top_catid]
#             print('MIN: {}/MAX: {}'.format(np.min(p_class), np.max(p_class)))

#             # uv.plot_results(x_test, x_test_im, sensMap, pred_diff, classifier.categories, top_catid)
#             # uv.plot_results(x_test, x_test_im, sensMap, pred_diff, classifier.categories, 385)
#             # uv.plot_results(x_test, x_test_im, sensMap, pred_diff, classifier.categories, 386)
#             print(y_pred_label, top_prob)
#             for cid in range(test_classes):
#                 catid = top_catids[0][cid].item()
#                 uv.plot_results(x_test, x_test_im, sensMap, pred_diff, classifier.categories, catid, show=True)
#         except:
#             print(y_pred_label, top_prob)

#         # for catid in np.random.randint(1000, size=test_classes):
#         #     uv.plot_results(x_test, x_test_im, sensMap, pred_diff, classifier.categories, catid)

        

def main():

    parser = argparse.ArgumentParser(description='Prediction Difference Analysis')
    parser.add_argument('-v', '--visualize',action="store_true", help='generate testbench files')
    parser.add_argument('-c', '--convert',action="store_true", help='generate testbench files')
    
    args = parser.parse_args()


    if MODEL == 'mnv2':
        model = infoMobileNetV2(show=False)
    elif MODEL == 'resnet50':
        # resnet versions: 18,34,50,101,152
        model = infoResNet(mVersion='resnet50')
    elif MODEL == 'alexnet':
        model = infoAlexNet()
    elif MODEL == 'vgg16':
        model = infoVgg16()
    elif MODEL == 'c_jakob':
        path = './model/ood/react/' + REGRESSION + 'resnet50' + '{}'.format(NAME) +  '{}.pth'.format(DSET)
        model = jml.htvc_resnet50(path,DSET,CLASSES)
    elif MODEL == 'c_conrad':
        path = './model/ood/isomax/'
        model = infoConradRSICD(path, dset=DSET)
    else:
        print('unknown model name', MODEL)
        return

    # if args.visualize:
    #     visualize(model, test_classes=5)
    #     return 
    # else: 
        
    experiment(model, args.visualize, args.convert)

    # viewImage(model)
    


if __name__=='__main__':
    print('troch', torch.__version__)
    print('numpy', np.version.version)
    main()

