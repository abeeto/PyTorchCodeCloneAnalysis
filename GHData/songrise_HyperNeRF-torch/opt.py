import argparse

def get_opts():
    parser = argparse.ArgumentParser()


    parser.add_argument('--root_dir', type=str,
                        default= "/root/autodl-tmp/data/fangzhou_nature",
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='llff',
                        choices=['blender', 'llff'],
                        help='which dataset to train/val')
    # parser.add_argument('--img_wh', nargs="+", type=int, default=[252,189],
    #                     help='resolution (img_w, img_h) of the image')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[270,480],
                        help='resolution (img_w, img_h) of the image')

    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--perturb', type=float, default=1.0,
                        help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='std dev of noise added to regularize sigma')
        
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse'],
                        help='loss to use')

    parser.add_argument('--batch_size', type=int, default=2048,
                        help='batch size')
    parser.add_argument('--chunk', type=int, default=2048,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--precision', type=int, default=16)

    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained model weight to load (do not load optimizers, etc)')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')
    ###########################
    #### params for poly ####
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')
    ###########################

    parser.add_argument('--exp_name', type=str, default='embed',
                        help='experiment name')

    ###########################
    #### params for warp ####
    parser.add_argument('--use_warp', type=bool, default=True,
                        help='whether to use warping, when set true, the warp embedding is also used')
    parser.add_argument('--slice_method', type=str, default='bendy_sheet',
                            help='method to slice the hyperspace, must be used with warping',
                            choices=['bendy_sheet', 'none', 'axis_aligned_plane'])
    parser.add_argument('--hyper_slice_out_dim', type=int, default=4,
                            help='The output dimension of the hypersheet mlp')
    parser.add_argument('--use_nerfies_meta',type=bool, default=True,
                help="whether to use the metadata (embeddings) for each rays")

    ###########################
    #### params for embedding ####
    parser.add_argument("--meta_GLO_dim",type=int,default=8,
                            help="the dimension used for GLO embedding of time")
    parser.add_argument("--share_GLO",type=bool,default=True,
                            help="When set true, all GLO embedding use the same model and key")
    parser.add_argument("--use_nerf_embedding",action="store_true",
                            help="whether to use the nerf embedding")
    parser.add_argument("--use_alpha_condition",action="store_true",
                            help="whether to use the alpha condition, must be used with use_nerf_embedding")
    parser.add_argument("--use_rgb_condition",action="store_true",
                            help="whether to use the rgb condition, must be used with use_nerf_embedding")                     

    parser.add_argument("--xyz_fourier",type=int,default=10,
                            help="the dimension used for fourier embedding of points xyz")
    parser.add_argument("--hyper_fourier",type=int,default=6,
                            help="the dimension used for fourier embedding of points hyper feature")
    parser.add_argument("--view_fourier",type=int,default=6,
                        help="the dimension used for fourier embedding of view dir ")
    return parser.parse_args()
