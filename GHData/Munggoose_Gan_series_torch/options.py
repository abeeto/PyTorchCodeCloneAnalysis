import argparse

class Option:
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--n_epochs",type=int,default=200, help = 'number of epochs of training')
        parser.add_argument("--batch_size",type=int, default=64, help = 'size of the batches')
        parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
        parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
        parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
        parser.add_argument("--device",type=str, default='cuda',help='device cuda or cpu')
        parser.add_argument("--ngf",type=int, default=128,help='number of generator featmap')
        parser.add_argument("--ndf",type=int, default=128,help='number of discriminator featmap')
        parser.add_argument("--channels", type=int, default=1, help="number of image channels")
        parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
        parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
        parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
        parser.add_arguemnt("--isTrain",type=bool, default=True, help= 'check is trainning mode')

        #Ganomaly option
        parser.add_argument("--w_adv",type=float,default=1, help = 'weight of discrminator loss')
        parser.add_argument("--w_con",type=float,default=1, help = 'weight of image loss')
        parser.add_argument("--w_enc",type=float,default=1, help = 'weight of feature loss')
        parser.add_argument('--abnormal_idx',type = int, default = 0,help= 'Abnomral')


        self.opt =parser.parse_args()
        print(self.opt)
    
    def parse(self):
        """ Parse Arguments.
        """
        args = vars(self.opt)
        return self.opt