from searcher.classifier import Classifier
import argparse

import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_dst", type = str, help = "The src directory for training.", required = False)
    parser.add_argument("-m", "--model_dir", type = str, help = "Model data directory", required = True)

    args = parser.parse_args()
    cls = Classifier(data_dir_path = args.src_dst)


    model_path = os.path.join(args.model_dir, "model")


    if os.path.exists(model_path):
        print("we are loading model")
        cls.load(model_dir = args.model_dir)


    cls.fit(epochs = 100, save_dir_path = args.model_dir)






