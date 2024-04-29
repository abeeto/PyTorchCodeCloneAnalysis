import time
import os
import argparse
from torch.utils.data import DataLoader
import torch


from TCAV.tcav import TCAVCompute
from model_torch import load_model
from utils import save_ace_report,plot_concepts
from ace_torch import ConceptDiscovery
from model_torch import load_model
from dataset_torch import ImageDataset



def discover_concepts(target_class,working_dir,source_dir,label_path,max_imgs,device,bottlenecks,num_workers,
                      random_dir,segment_method,cluster_method):
    """discover concept using segmentation and clustering methods"""
    # create working dirs
    current_working_name = time.strftime("%Y%m%d_%H%M_%S", time.localtime()) + "_" + target_class
    current_working_dir = os.path.join(args.working_dir,current_working_name)
    discovered_concepts_dir = os.path.join(current_working_dir, 'concepts/')
    results_dir = os.path.join(current_working_dir, 'results/')
    cavs_dir = os.path.join(current_working_dir, 'cavs/')
    activations_dir = os.path.join(current_working_dir, 'acts/')
    results_summaries_dir = os.path.join(current_working_dir, 'results_summaries/')

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    os.makedirs(current_working_dir)
    os.makedirs(discovered_concepts_dir)
    os.makedirs(results_dir)
    os.makedirs(cavs_dir)
    os.makedirs(activations_dir)
    os.makedirs(results_summaries_dir)
    
    img_dir = os.path.join(discovered_concepts_dir, 'images')
    os.makedirs(img_dir)
     

    image_dataset = ImageDataset(source_dir,target_class,label_path,max_imgs)
    img_loader = DataLoader(image_dataset,batch_size=max_imgs,shuffle=True,num_workers=20)
    device = torch.device(device)
    model = load_model("googlenet",bottlenecks.split(',')).to(device)  # 将模型写入到GPU中去

    concept_discovery = ConceptDiscovery(model,
                                        img_loader,
                                        target_class,
                                        bottlenecks=bottlenecks.split(','),
                                        device=device,
                                        num_workers=num_workers)
    
    param_dict = {'n_clusters': 25,
                'n_segments': [15, 50, 80]}
    concept_discovery.construct_concept(img_dir,discovered_concepts_dir,random_dir,segment_method,
        cluster_method,param_dict)
    print('success for constructing concepts!')
    
    return current_working_dir,model
    
    
def get_tcav(result_dir,layers,model=None,save=True,plot=True):
    """if necessary, calculate tcav value and plot images"""
    concept_dir = 'concepts'
    cav_dir = 'cavs'
    activation_path = os.path.join(result_dir,concept_dir,'concept_info.npy')
    random_path = os.path.join(result_dir,concept_dir,'random_info.npy')

    if (not os.path.exists(activation_path)) or (not os.path.exists(random_path)):
        raise "No activation and random file available, please discover concepts first"
    if model is None:
      model = load_model("googlenet",layers)
    tcav = TCAVCompute(model,layers,activation_path,random_path)
    scores = tcav.interpret(force_train=True,weight_path=os.path.join(result_dir,cav_dir),
        save_path=os.path.join(result_dir,cav_dir),save=True)

    if save:
        address = os.path.join(result_dir,"results_summaries/ace_results.txt")
        save_ace_report(scores,address)
    if plot:
        address_img = os.path.join(result_dir,"results/")
        plot_concepts(concept_dict=tcav.train_concept.concept_dict,scores=scores,layers=layers,address=address_img)
    print('success for calculating tcav!')

    
def main(args):
    # step1: discover concepts
    if args.discover_concepts:

      current_working_dir,model = discover_concepts(target_class=args.target_class,working_dir=args.working_dir,
          source_dir=args.source_dir,label_path=args.label_path,max_imgs=args.max_imgs,
          device=args.device,bottlenecks=args.bottlenecks,num_workers=args.num_workers,
          random_dir=args.random_dir,segment_method=args.segment_method,cluster_method=args.cluster_method)
    
      # step2: calculate tcav
      get_tcav(result_dir=current_working_dir,layers=args.bottlenecks.split(','),model=model)
    
    else:
        assert os.path.exists(args.current_working_dir),"no working directory available, please check it"

        get_tcav(result_dir=args.current_working_dir,layers=args.bottlenecks.split(','),save=args.save_txt,plot=args.save_concepts)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', type=str,
      help='Directory to save the results.', default='/data/aaron/adversarial_attack/explain_methods/ACE_PyTorch_implementation/ACE_data/ACE_torch')
    parser.add_argument('--source_dir', type=str,
      help='Directory where the networks classes image folders', default='/data/dataset/ImageNet2012/train')
    parser.add_argument('--random_dir', type=str,
      help='Directory where the networks random image folders', default='/data/aaron/adversarial_attack/explain_methods/ACE_PyTorch_implementation/ACE_data/ImageNet')
    parser.add_argument('--label_path', type=str,
      help='Path to model checkpoints.', default='/data/dataset/ImageNet2012/imagenet_labels.csv')
    parser.add_argument('--target_class', type=str,
      help='The name of the target class to be interpreted', default='zebra')
    parser.add_argument('--max_imgs', type=int,
      help="Maximum number of images in a discovered concept",default=40)
    parser.add_argument('--num_workers', type=int,
      help="Number of parallel jobs.",default=20)
    parser.add_argument('--bottlenecks', type=str,
      help='Names of the target layers of the network (comma separated)',default='inception4c')  
    parser.add_argument('--device', type=str,
      help='cuda:0,cuda:1 or cpu',default='cuda:0')  
    parser.add_argument('--segment_method', type=str,
      help='segment method',default='slic')
    parser.add_argument('--cluster_method', type=str,
      help='cluster method',default='KM')
    parser.add_argument('--discover_concepts',type=bool,
        help='choose whether to discover concepts by ACE method',default=True)
    parser.add_argument('--current_working_dir',type=str,
        help='current result directory for saving ACE results',default=None)
    parser.add_argument('--save_txt',type=bool,
        help='whether to save txt data for calculating tcav value',default=True)
    parser.add_argument('--save_concepts',type=bool,
        help='whether to save and plot image for concepts',default=True)
      
    
    args = parser.parse_args()
    
    
    main(args)