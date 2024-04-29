import re
import torch
import pathlib
import importlib
from typing import Union

# Configurable options
MODEL_FILE = "model.py"
STATE_FILE = "squeezenet1_1-b8a52dc0.pth"
OUTPUT_PATH = "output.onnx"

def get_first_class(module_path):
    with open(module_path, "r") as model_file:
        lines = model_file.readlines()

    class_lines = [line for line in lines if "class" in line]

    if len(class_lines) == 0:
        raise Exception("No class.")

    first_class = re.sub(r'[\s+:]', '', class_lines[0].replace("class", ""))

    if "(" in first_class:
        first_class = first_class.split("(")[0]

    return first_class

def get_module_name(module_path: str):
    last_item = module_path.split("/")[-1]
    return last_item.split(".")[0]

def convert_pytorch_to_onnx(model_file: Union[str, pathlib.Path],
                            state_file: Union[str, pathlib.Path], 
                            output_path: str):
    """
    Converts a given PyTorch model into the ONNX format.
    """
    module_name = get_module_name(model_file)
    model_class_name = get_first_class(model_file)
    ModelClass = getattr(importlib.import_module(module_name), model_class_name)
    
    model = ModelClass()
    model.load_state_dict(torch.load(state_file))
    # Evaluate the model to switch some operations from training mode to inference.
    model.eval()
    # Create dummy input for the model. It will be used to run the model inside export function.
    dummy_input = torch.randn(1, 3, 224, 224)
    # Call the export function
    torch.onnx.export(model, (dummy_input, ), output_path)


# Invoking the converter
if __name__ == "__main__":
    convert_pytorch_to_onnx(model_file=MODEL_FILE, state_file=STATE_FILE, output_path=OUTPUT_PATH)