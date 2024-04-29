import os
import torch
import torchvision
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from utils.utils import get_device, visualize
from Data.Dataset import Detection
import sys
from tqdm import tqdm
from torchvision import transforms
from pathlib import Path

def print_help():
    """
    Print instruction to the standard output
    """
    print("usage: python eval.py [options]")
    print('options:')
    print("-d <argument>, --directory <argument>       Path to the folder where images are stored")
    print("-f <argument>, --file <argument>            Path to the file")
    print("User must provide one of two options mentioned above")
    print("-a <argument>, --architecture <argument>    Name of used architecture, possible choices: unet, unet++, deeplab and deeplab+")
    print("-e <argument>, --encoder <argument>         Name of the encoder, possible choices: resnet50, resnext50 and efficientnet")
    print("-s <argument>, --size <argument>            Size of image during detection, possible choices: 512, 768 and 1024")
    print("-w <argument>, --weights <argument>         Path to the zip file containing model weights. They must match model defined using other parameters")
    print("-o <argument>, --output <argument>          Path to the folder where results will be stored, if not provided script location will be used")
    print("Please note that all predictions will be saved in given output directory as prediction_[original file name]")
    print("ex. python eval.py -a unet -e efficientnet -s 1024 -w model.zip -f demo.jpg -o img/")


def validate_size(val):
    if val in ["512", "768", "1024"]:
        return int(val)
    else:
        sys.exit('Undefined image size declared')

def validate_encoder(val):
    if val=="resnet50":
        return "resnet50"
    elif val=="resnext50":
        return "resnext50_32x4d"
    elif val=="efficientnet":
        return "efficientnet-b4"
    else:
        sys.exit('Undefined encoder declared')


def get_model(val, enc):
    model=None
    if val=="unet":
        model = smp.Unet(
            encoder_name=enc,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=10,  # model output channels (number of classes in your dataset)
        )
    elif val=="unet++":
        model = smp.UnetPlusPlus(
            encoder_name=enc,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=10,  # model output channels (number of classes in your dataset)
        )
    elif val=="deeplab":
        model = smp.DeepLabV3(
            encoder_name=enc,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=10,  # model output channels (number of classes in your dataset)
        )
    elif val=="deeplab+":
        model = smp.DeepLabV3Plus(
            encoder_name=enc,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=10,  # model output channels (number of classes in your dataset)
        )
    else:
        sys.exit('Undefined network architecture declared')
    return model
def main(argv):
    output_dir=""
    enc=None
    s=None
    w=None
    arch=None
    inp=None
    #opts, args = getopt.getopt(argv, "hd:f:aeswo:", ["help=", "directory=", "file=", "architecture", "encoder", "size", "weights", "output="])
    opts=[*zip(argv[::2], argv[1::2])]
    if "-h" in argv or "--help" in argv:
        print_help()
        sys.exit()
    for opt, arg in opts:
        if opt in ("-o", "--output"):
            output_dir = arg
        elif opt in ("-e", "--encoder"):
            enc=validate_encoder(arg)
        elif opt in ("-s", "--size"):
            s=validate_size(arg)
        elif opt in ("-w", "--weights"):
            w=arg
        elif opt in ("-a", "--architecture"):
            arch=arg
        elif opt in ("-f", "--file", "-d", "--directory"):
            inp=arg
    try:
        model=get_model(arch, enc)
        transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((s, s), interpolation=torchvision.transforms.InterpolationMode.NEAREST),
                                         ])
        DEVICE=get_device()
        if torch.cuda.is_available():
            model.cuda()
        try:
            model.load_state_dict(torch.load(w))
        except Exception:
            sys.exit("Could not load given weights")

        if os.path.isdir(inp):
            files=[os.path.join(r,file) for r,d,f in os.walk(inp) for file in f]
        else:
            if os.path.exists(inp):
                files=[inp]
            else:
                sys.exit("Given file does not exist")
        test_dataset = Detection(files, transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    except UnboundLocalError as e:
        print(e)
        sys.exit("Not all arguments were defined, use --help for more details")
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        for (i, (x, y)) in tqdm(enumerate(test_loader)):
            # send the input to the device
            (x, y) = (x.to(DEVICE), y.to(DEVICE))
            # make the predictions and calculate the validation loss
            pred = model(x)
            pred = torch.sigmoid(pred)
            head=files[i].split("/")[-1]
            filename = Path(output_dir, "prediction_"+head)
            visualize(filename, Image=x[0].cpu().data.numpy(),
                            Prediction=pred.cpu().data.numpy()[0].round())
if __name__ == "__main__":
    main(sys.argv[1:])