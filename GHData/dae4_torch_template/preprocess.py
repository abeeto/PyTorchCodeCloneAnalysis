import os
import shutil
from distutils.dir_util import copy_tree
# backup 
def backup(args):
    if not os.path.exists(args["backDir"]):
        print("Checkpoint Directory does not exist! Making directory {}".format(args["backDir"]))
        os.makedirs(args["backDir"], exist_ok=True)
    else:
        shutil.rmtree(args["backDir"])
        os.makedirs(args["backDir"], exist_ok=True)
    filenames = os.listdir("./")
    for name in filenames:
        if "model_" in name:
            model_dir = name
    ## model
    shutil.move(model_dir,'backup')
    ## data
    copy_tree('src','backup/')

def update(args):
    root = args["update"]
    newDir = root+'/new'
    delDir = root+'/delete'
    existDir = root+'/exist'
    ## new
    if newDir:
        new_class = os.listdir(newDir)
    ## delete
    elif delDir:
        del_class = os.listdir(delDir)
    ## exist
    elif existDir:
        exist_class = os.listdir(existDir)
