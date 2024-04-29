import argparse
import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, default='WAAE', help='model')
parser.add_argument('--architecture', type=str, default='ResNet', help='model architecture')
parser.add_argument('--k', type=int, default=5, help='times critic is trained per one iteration of generator training')
parser.add_argument('--b1', type=float, default=0.50, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.997, help='adam: decay of first order momentum of gradient')
parser.add_argument('--batch_size', type=int, default=16, help='size of the batches')
parser.add_argument('--n_channels', type=int, default=3, help='number of image channels')
parser.add_argument('--folder_name', type=str, default='results_26May_04', help='name of the folder to save the files')
parser.add_argument('--latent_dim', type=int, default=30, help='dimensionality of the latent code')
parser.add_argument('--lr_R', type=float, default=0.00004, help='reconstruction learning rate')
parser.add_argument('--lr_D', type=float, default=0.00004, help='discriminator learning rate')
parser.add_argument('--lr_G', type=float, default=0.00002, help='generator learning rate')
parser.add_argument('--momentum', type=float, default=0.8, help='batch normalisation momentum')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--n_epochs', type=int, default=10000, help='number of epochs of training')
parser.add_argument('--sample_interval', type=int, default=100, help='interval between image sampling')
parser.add_argument('--s_sd', type=int, default=1.0, help='standard deviation for sampling')
parser.add_argument('--n_sd', type=int, default=0.1, help='standard deviation for noise added to discriminator inputs')
parser.add_argument('--lr_step', type=int, default=50, help='adam: learning rate')
parser.add_argument('--lr_gamma', type=float, default=0.5, help='adam: learning rate')
parser.add_argument('--real', type=float, default=0.9, help='value for real label')
parser.add_argument('--fake', type=float, default=-0.9, help='value for fake label')
parser.add_argument('--h_dim', type=int, default=512, help='number of hidden units')
parser.add_argument('--img_size', type=int, default=64, help='size of each image dimension')
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for discriminator weights")
args = parser.parse_args()

# make folders to save results
os.makedirs('{}'.format(args.folder_name), exist_ok=True)

# open text file with append mode
file = open('models.txt', 'a')
file.write('' + '\n' + str(datetime.datetime.now()) + '\n' + str(args) + '\n')
file.close()