import argparse
import random
import timeit

from PIL import Image
from torch.backends import cudnn
import torch.utils.data.distributed
from torchvision import transforms, utils


from cyclegan_pytorch import Generator

parser = argparse.ArgumentParser(
    description="PyTorch implements `Unpaired Image-to-Image "
    "Translation using Cycle-Consistent Adversarial Networks`"
)
parser.add_argument(
    "--file",
    type=str,
    default="assets/horse.png",
    help="Image name. (default:`assets/horse.png`)",
)
parser.add_argument(
    "--model-name",
    type=str,
    default="weights/horse2zebra/netG_A2B.pth",
    help="dataset name.  (default:`weights/horse2zebra/netG_A2B.pth`).",
)
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument(
    "--image-size",
    type=int,
    default=256,
    help="size of the data crop (squared assumed). (default:256)",
)
parser.add_argument(
    "--manualSeed",
    type=int,
    help="Seed for initializing training. (default:none)",
)

args = parser.parse_args()
print(args)

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print(
        "WARNING: You have a CUDA device, so you "
        "should probably run with --cuda"
    )

device = torch.device("cuda:0" if args.cuda else "cpu")

# create model
model = Generator().to(device)

# Load state dicts
model.load_state_dict(torch.load(args.model_name))

# Set model mode
model.eval()

# Load image
image = Image.open(args.file)
pre_process = transforms.Compose(
    [
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)
image = pre_process(image).unsqueeze(0)
image = image.to(device)

start = timeit.default_timer()
fake_image = model(image)
elapsed = timeit.default_timer() - start
print(f"cost {elapsed:.4f}s")
utils.save_image(fake_image.detach(), "result.png", normalize=True)
