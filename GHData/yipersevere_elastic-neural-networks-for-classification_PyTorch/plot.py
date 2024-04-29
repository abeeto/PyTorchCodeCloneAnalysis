import matplotlib

matplotlib.use("PDF")
import matplotlib.pyplot as plt
plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams["figure.figsize"] = (11,5)

import numpy as np
import pandas as pd
import os
from cycler import cycler


from plot_hepler import mobileNet_cifar100, mobileNet_cifar10,\
                        inceptionv3_cifar10, inceptionv3_cifar100,\
                        pytorch_cifar10_loss, pytorch_ResNet,\
                        pytorch_DenseNet


def plot_error_fig(errors, layer_index, strDict):
    fig, ax = plt.subplots(1, sharex=True)
    colormap = plt.cm.tab20
    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 1, len(errors))])
    
    last = len(errors)-1
    elastic_last = len(errors)-2

    for k in range(len(errors)):
        # Plots
        x = np.arange(len(errors[k])) + 1
        if k == elastic_last:
            c_label = strDict["elastic_final_layer_label"]
        elif k == last:
            c_label = strDict["original_layer_label"]
        # elif k == (last+1):
        #     c_label = strDict["note_comment"]
        else:
            c_label = strDict["elastic_intermediate_layer_label"] + str(layer_index[k])
        ax.plot(x, errors[k], label=c_label)
        # Legends
        y = k
        x = len(errors)
        # ax.text(x, y, "%d" % k)
    ax.set_ylabel(strDict["y_label"])
    ax.set_xlabel(strDict["x_label"])
    ax.set_title(strDict["fig_title"])
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    fig_size = plt.rcParams["figure.figsize"]

    plt.rcParams["figure.figsize"] = fig_size

    plt.tight_layout()

    plt.savefig(strDict["save_file_name"], bbox_inches="tight")
    fig.savefig(strDict["save_png_file_name"], bbox_inches="tight")
    plt.close("all")


def plot_model_accuracy_on_CIFAR(dataframe, save_path, data):
    """
    plot star image in CIFAR with differnt models
    """
    # red dashes, blue squares and green triangles
    plt.style.use('seaborn-colorblind')
    fig = plt.figure(figsize=(5,4))
    ax = plt.subplot(111)
    ax.set_prop_cycle(cycler('color', ['yellowgreen','mediumseagreen','cadetblue','lightslategray','skyblue','darkslategray','yellowgreen','mediumseagreen','cadetblue','lightslategray','skyblue','darkslategray','gold']))

    for index, row in dataframe.iterrows():
        if "Elastic" in row["model"]:
            ax.semilogx(row["total floating point operations"], row["error"], '*', label=row["model"])
        else:
            ax.semilogx(row["total floating point operations"], row["error"],  markevery=2, drawstyle='steps', marker='o', markersize=3,linewidth=1, label=row["model"])
    plt.xlabel('FLOPs')
    plt.ylabel('Error (%)')

    plt.title("Classification on " + data)
    plt.legend(loc='upper right', prop={'size':6.4})
    # plt.tight_layout(pad=4)
    ax.grid(linestyle='--', linewidth=0.5)
    fig.savefig(save_path+ os.sep +'flops-' + data + '-accuracy-all-models-no-imagenet.pdf')
    fig.savefig(save_path+ os.sep +'flops-' + data + '-accuracy-all-models-no-imagenet.png')

def plot_ResNet_model_accuracy_on_CIFAR(dataframe, save_path, data):
    # red dashes, blue squares and green triangles
    plt.style.use('seaborn-colorblind')
    fig = plt.figure(figsize=(6,5))
    ax = plt.subplot(111)
    ax.set_prop_cycle(cycler('color', ['yellowgreen','mediumseagreen','cadetblue','lightslategray','skyblue','darkslategray','yellowgreen','mediumseagreen','cadetblue','lightslategray','skyblue','darkslategray','gold']))

    for index, row in dataframe.iterrows():
        if "Elastic" in row["model"]:
            ax.plot(row["layers"], row["error"], '*', label=row["model"])
        else:
            ax.plot(row["layers"], row["error"],  markevery=2, drawstyle='steps', marker='o', markersize=3,linewidth=1, label=row["model"])
    plt.xlabel('model-Layers')
    plt.ylabel('Error (%)')

    plt.title("Classification on " + data)
    plt.legend(loc='upper left', prop={'size':5.6})
    # plt.tight_layout(pad=4)
    ax.grid(linestyle='--', linewidth=0.5)
    fig.savefig(save_path+ os.sep +'layers-' + data + '-accuracy-all-ResNet-models.pdf')
    fig.savefig(save_path+ os.sep +'layers-' + data + '-accuracy-all-ResNet-models.png')

def plot_epochs_models_error(dataframe, save_path, data):
    """
    plot training, testing epoches error evolution for all different models on specific dataset
    """

    plt.style.use('seaborn-colorblind')
    fig = plt.figure(figsize=(5,4))
    ax = plt.subplot(111)
    save_path = "/home/yi/Dropbox/aicas2019/AICAS-LaTeX-template/plot"
    fig.savefig(save_path+ os.sep + '-accuracy-all-models-no-imagenet.pdf')
    model = None

    with_Elastic = None
    without_Elastic = None

    if data == "error":
        y_label_str = "error(%)"
    elif data == "loss":
        y_label_str = "loss"

    if data == "CIFAR10" or data == "cifar10":
        fig_title_str = "classification error on CIFAR-10"

        if model == "DenseNet121" or model == "densenet121":
            # folder = "/home/yi/narvi/elastic/pytorch_code/Elastic_ResNet18/Classification_Accuracy/"
            # origin_file = folder + os.sep + "pytorch_CIFAR10_0_intermediate_classifiers_Elastic_ResNet18_include_pretrain_skip_lastCLF/2018-07-10-16-59-37/test_errors.txt"
            # elastic_file = folder + os.sep + "pytorch_CIFAR10_all_intermediate_classifiers_Elastic_ResNet18_include_pretrain_skip_lastCLF/2018-07-10-16-59-37/test_errors.txt"
            # fig_title_prefix = "Elastic ResNet 18 "
            # save_file_name = "Pytorch_CIFAR_10_accuracy_Elastic&Original_ResNet18_include_pretrain_skip_last_interCLF"
            # # 这里是[0,7] 而不是[0,8], 即把最后一层的intermediate classifier跳过了
            # layer_plot_index = [0,1,2,3,4,5,6,7]
            # original_layer_label = "Original_ResNet-18"   

            folder = "/home/yi/narvi/elastic/pytorch_code/Elastic_DenseNet121/Classification_Accuracy"
            origin_file = folder + os.sep + "pytorch_CIFAR10_0_intermediate_Elastic_DenseNet121_include_pretrain_skip_last_interCLF/2018-07-23-16-07-41/test_errors.txt"
            elastic_file = folder + os.sep + "pytorch_CIFAR10_all_intermediate_Elastic_DenseNet121_include_pretrain_skip_last_interCLF/2018-07-23-16-05-30/test_errors.txt"
            fig_title_prefix = "Elastic DenseNet 121  "
            save_file_name = "Pytorch_CIFAR_10_accuracy_Elastic&Original_DenseNet-121"
            # 这里是[0,7] 而不是[0,8], 即把最后一层的intermediate classifier跳过了
            layer_plot_index = [0,1,2,3]
            original_layer_label = "Original_DenseNet-121"   

        elif model == "DenseNet169" or model == "densenet169":
            folder = "/home/yi/narvi/elastic/pytorch_code/Elastic_DenseNet169/Classification_Accuracy"
            origin_file = folder + os.sep + "pytorch_CIFAR10_0_intermediate_Elastic_DenseNet169_include_pretrain_skip_last_interCLF/2018-07-24-00-06-26/test_errors.txt"
            elastic_file = folder + os.sep + "pytorch_CIFAR10_all_intermediate_Elastic_DenseNet169_include_pretrain_skip_last_interCLF/2018-07-24-00-06-24/test_errors.txt"
            fig_title_prefix = "Elastic DenseNet 169 "
            save_file_name = "Pytorch_CIFAR_10_accuracy_Elastic&Original_DenseNet-169"
            layer_plot_index = [0,1,2,3]
            original_layer_label = "Original_DenseNet-169"   

        else:
            NotImplementedError

    elif data == "CIFAR100" or data == "cifar100":
        fig_title_str = fig_title_str_cifar100
        NotImplementedError   
    else:
        print("data should be CIFAR 10 or CIFAR 100")
        NotImplementedError

    
    error_origin = pd.read_table(origin_file, delim_whitespace=True, header=None)
    error_elastic = pd.read_table(elastic_file, delim_whitespace=True, header=None) 
    
    save_folder = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/Elastic/elastic/pytorch_code/plot"

    captionStrDict = {
        "save_file_name" : save_folder + os.sep + save_file_name +".pdf",
        "save_png_file_name" : save_folder + os.sep + save_file_name +".png",
        "fig_title" : fig_title_prefix + fig_title_str,
        "x_label" : x_label_str,
        "y_label" : y_label_str,
        'elastic_final_layer_label': "Final_Output_Classifier",
        "elastic_intermediate_layer_label" : "Intermediate_Classifier_",
        "original_layer_label" : original_layer_label
    }
    return with_Elastic, without_Elastic, captionStrDict    



if __name__ == "__main__":
    
    errors = []


    # error_origin, error_elastic, layer_plot_index, captionStrDict = pytorch_ResNet(model="ResNet18", data="CIFAR100")
    # error_origin, error_elastic, layer_plot_index, captionStrDict = pytorch_ResNet(model="ResNet50", data="CIFAR100")
    # error_origin, error_elastic, layer_plot_index, captionStrDict = pytorch_ResNet(model="ResNet152", data="CIFAR100")
    
    error_origin, error_elastic, layer_plot_index, captionStrDict = pytorch_DenseNet(model="DenseNet169", data="CIFAR10")
    
    # error_origin, error_elastic, layer_plot_index, captionStrDict = resetnet152_cifar100()
    
    # error_origin, error_elastic, layer_plot_index, captionStrDict = resetnet152_cifar10()
    # error_origin, error_elastic, layer_plot_index, captionStrDict = resetnet152_cifar100()

    # error_origin, error_elastic, layer_plot_index, captionStrDict = resetnet50_cifar100()
    # error_origin, error_elastic, layer_plot_index, captionStrDict = resetnet50_cifar10()
    
    # error_origin, error_elastic, layer_plot_index, captionStrDict = inceptionv3_cifar10()
    # error_origin, error_elastic, layer_plot_index, captionStrDict = inceptionv3_cifar100()
    

    # error_origin, error_elastic, layer_plot_index, captionStrDict = mobileNet_cifar100()
    # error_origin, error_elastic, layer_plot_index, captionStrDict = mobileNet_cifar10()
    
    # error_origin, error_elastic, layer_plot_index, captionStrDict = pytorch_cifar10()
    # error_origin, error_elastic, layer_plot_index, captionStrDict = pytorch_cifar10_loss()
    
    # result_df = pd.DataFrame()
    # result_df = add_evaluation_columns_df(result_df, error_elastic, error_origin, "MobileNets_alpha_0.75", "error")
    
    # f1_origin, f1_elastic, f1_layer_plot_index, f1_captionStrDict = mobileNets_alpha_0_75_F1_cifar100()
    # result_df = add_evaluation_columns_df(result_df, f1_elastic, f1_origin, "MobileNets_alpha_0.75", "f1_score")

    # MobileNets_alpha_0_75_FLOPs_ConvLayer = [22579200, 14450688, 28901376, 14450688, 28901376, 14450688, 28901376, 28901376, 28901376, 28901376, 28901376, 14450688, 28901376]
    # MobileNets_alpha_0_75_FLOPs_OutputLayer = [4848, 9696, 9696, 19392, 19392, 38784, 38784, 38784, 38784, 38784, 38784, 77568, 77568, 3800832]
    # MobileNets_alpha_0_75_FLOPs_block_Conv_Output = [22584048, 14460384, 28911072, 14470080, 28920768, 14489472, 28940160, 28940160, 28940160, 28940160, 28940160, 14528256, 28978944, 3800832]
    # MobileNets_alpha_0_75_FLOPs_Cumulative = [22584048, 37044432, 65955504, 80425584, 109346352, 123835824, 152775984, 181716144, 210656304, 239596464, 268536624, 283064880, 312043824, 315844656]




    # # best_acc_df = add_criteria_columns_df(best_acc_df, error_elastic, error_origin, "InceptionV3", "F1 score")


    # result_df.to_json("result_Inception.json")


    for i in layer_plot_index:
        errors.append(list(error_elastic.iloc[:, i]))
    
    errors.append(list(error_origin.iloc[:,0]))

    plot_error_fig(errors, layer_plot_index, captionStrDict)

    # cifar_10_model_result = pd.read_csv("/media/yi/e7036176-287c-4b18-9609-9811b8e33769/Elastic/elastic/pytorch_code/plot/cifar10_ResNet_model_result_in_pytorch.csv")
    # plot_ResNet_model_accuracy_on_CIFAR(cifar_10_model_result, "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/Elastic/elastic/pytorch_code/plot", data="CIFAR-10")

    # cifar_100_model_result = pd.read_csv("/media/yi/e7036176-287c-4b18-9609-9811b8e33769/Elastic/elastic/pytorch_code/plot/cifar100_ResNet_model_result_in_pytorch.csv")
    # plot_ResNet_model_accuracy_on_CIFAR(cifar_100_model_result, "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/Elastic/elastic/pytorch_code/plot", data="CIFAR-100")


