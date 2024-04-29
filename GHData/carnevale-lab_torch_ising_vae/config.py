import json
from pathlib import Path


class Config:

    def __init__(self):
        # reasonable defaults
        self.config = {
            "BATCH_SIZE": 50,
            "SEQ_LENGTH": 42,
            "ALPHABET_LENGTH": 21,
            "NUM_NODES": 512,
            "EMBEDDING_DIM": 4,
            "EPOCHS": 10,
            "OUTPUT_PATH": "model",
            "DO_LAYERS": None,
            "BN_LAYERS": None,
        }

    def __getattr__(self, item):
        if item == "config":
            return super().__getattribute__(item)
        return self.config[item]

    def __setattr__(self, key, value):
        if key == "config":
            return super().__setattr__(key, value)
        self.config[key] = value

    def __setitem__(self, key, value):
        if key == "config":
            return super().__setitem__(key, value)
        self.config[key] = value

    def __getitem__(self, item):
        if item == "config":
            return super().__getitem__(item)
        return self.config[item]

    @classmethod
    def from_json_file(cls, file: Path):
        instance = Config()
        with file.open('r') as fp:
            js = json.load(fp)
            for key in js.keys():
                instance[key] = js[key]

        return instance
