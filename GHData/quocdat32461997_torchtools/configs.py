import os
import json
import torch


class Configs(object):
    def __init__(self, path=None, args=None):
        self.args = args
        if path is not None:
            assert os.path.exists(path) and path.endswith("json")
            with open(path, "r") as file:
                self.from_json(json.load(file))
        elif args is not None:
            assert isinstance(args, dict) is True
            self.from_json(args)
        else:
            raise Exception("Either path or args must be a valid value")

        if not self.devices:
            self.devices = "auto"
        self.accelerator = "cuda" if torch.cuda.is_available() else "cpu"

    def from_json(self, configs):
        self.__dict__.update(configs)
        self.args = {k: v for k, v in self.__dict__.items()}  # save args

        # create saved_dir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def to_json(self, path):
        assert isinstance(path, str) and path.endswith(".json")

        with open(path, "w") as file:
            json.dump(self.args, file)
