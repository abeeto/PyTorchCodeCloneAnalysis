import torch
import torchvision

if __name__ == "__main__":
    model = torch.load('backup/lights.weights')
    dummy_input = torch.randn(16, 3, 416, 416)

    # Providing input and output names sets the display names for values
    # within the model's graph. Setting these does not change the semantics
    # of the graph; it is only for readability.
    #
    # The inputs to the network consist of the flat list of inputs (i.e.
    # the values you would pass to the forward() method) followed by the
    # flat list of parameters. You can partially specify names, i.e. provide
    # a list here shorter than the number of inputs to the model, and we will
    # only set that subset of names, starting from the beginning.
    input_names = [ "actual_input_1" ]
    output_names = [ "output1" ]

    torch.onnx.export(model, dummy_input, "yolo.onnx", verbose=True, input_names=input_names, output_names=output_names)

