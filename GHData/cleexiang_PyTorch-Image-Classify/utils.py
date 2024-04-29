import torch
import torchvision
import coremltools as ct

def convert_to_ml():
    torch_model = torch.load("model.pth")
    torch_model.eval()
    example_input = torch.rand(1, 3, 224, 224)
    traced_model = torch.jit.trace(torch_model, example_input)
    # Using image_input in the inputs parameter:
    # Convert to Core ML using the Unified Conversion API.
    model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=example_input.shape)]
    )
    # Save the converted model.
    model.save("mobilenet.mlmodel")

    # # Load a pre-trained version of MobileNetV2
    # torch_model = torchvision.models.mobilenet_v2(pretrained=True)
    # # Set the model in evaluation mode.
    # torch_model.eval()
    #
    # # Trace the model with random data.
    # example_input = torch.rand(1, 3, 224, 224)
    # traced_model = torch.jit.trace(torch_model, example_input)
    # out = traced_model(example_input)
    # # Using image_input in the inputs parameter:
    # # Convert to Core ML using the Unified Conversion API.
    # model = ct.convert(
    #     traced_model,
    #     inputs=[ct.TensorType(shape=example_input.shape)]
    # )
    # # Save the converted model.
    # model.save("mobilenet.mlmodel")

if __name__ == '__main__':
    convert_to_ml()

