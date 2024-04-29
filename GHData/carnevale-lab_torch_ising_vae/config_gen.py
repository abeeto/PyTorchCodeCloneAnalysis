from config import Config
from pathlib import Path
from copy import deepcopy
import json

conf = Config.from_json_file(Path("gen_config.json"))

data = {}
dicts = []
c=0
for epoch_count in conf.epochs:
    data['EPOCHS'] = epoch_count

    for batch_size in conf.batch:
        data['BATCH_SIZE'] = batch_size

        for hidden_layers in conf.layers:
            data['LAYERS'] = hidden_layers
            data['IN_DIM'] = 900

            for num_nodes in conf.nodes:
                data['NUM_NODES'] = num_nodes

                for latents in conf.latent:
                    data['LATENT'] = latents

                    for activate in conf.activation:
                        data['ACTIVATE'] = activate
                        data['LPLOT'] = 0
                        data['TSNE'] = 0
                        data['HAMMING'] = 0
                        data['COVARS'] = 0
                        if(num_nodes < latents):
                            continue
                        dicts.append(deepcopy(data))
                        c+=1



conf_list = []
for i in range(len(dicts)):
    outname = "config_E" + str(dicts[i]["EPOCHS"]) + "_B" + str(dicts[i]["BATCH_SIZE"]) + "_D" + str(dicts[i]["LAYERS"]) + "_N" + str(dicts[i]["NUM_NODES"]) + "_L" + str(dicts[i]["LATENT"]) + "_" + dicts[i]["ACTIVATE"] + ".json"
    conf_list.append(deepcopy(outname))
    with open("configs/" + outname, 'w') as outfile:
        json.dump(dicts[i], outfile)


f1= open("commands1", "w")
for i in range(270):
    f1.write("python pytorch_vae.py " + conf_list[i] + " loss1.txt" + '\n')

f2 = open("commands2", "w")
for i in range(270):
    f2.write("python pytorch_vae.py " + conf_list[i+270] + " loss2.txt" + '\n')

f3 = open("commands3", "w")
for i in range(270):
    f3.write("python pytorch_vae.py " + conf_list[i+540] + " loss3.txt" + '\n')

f4 = open("commands4", "w")
for i in range(270):
    f4.write("python pytorch_vae.py " + conf_list[i+810] + " loss4.txt" + '\n')







                    
                    
