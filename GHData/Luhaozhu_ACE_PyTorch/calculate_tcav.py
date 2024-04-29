import numpy as np
import os
from TCAV.tcav import TCAVCompute
from model_torch import load_model
from utils import save_ace_report,plot_concepts


def get_tcav(result_dir,layers,save=True,plot=True):
    concept_dir = 'concepts'
    cav_dir = 'cavs'
    activation_path = os.path.join(result_dir,concept_dir,'concept_info.npy')
    random_path = os.path.join(result_dir,concept_dir,'random_info.npy')


    model = load_model("googlenet",layers)
    tcav = TCAVCompute(model,layers,activation_path,random_path)
    scores = tcav.interpret(force_train=False,weight_path=os.path.join(result_dir,cav_dir),
        save_path=os.path.join(result_dir,cav_dir),save=True)

    print(scores)
    if save:
        address = os.path.join(result_dir,"results_summaries/ace_results.txt")
        save_ace_report(scores,address)
    if plot:
        address_img = os.path.join(result_dir,"results/")
        plot_concepts(concept_dict=tcav.train_concept.concept_dict,scores=scores,layers=layers,address=address_img)

if __name__ == "__main__":
    result_dir = "/data/aaron/ACE/ACE_torch/20221109_1001_22_zebra"
    layers = ['inception4c']
    get_tcav(result_dir,layers)
