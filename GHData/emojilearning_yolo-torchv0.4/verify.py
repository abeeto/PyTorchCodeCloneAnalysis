import onnx

if __name__ == "__main__":
    # Load the ONNX model
    model = onnx.load("yolo.onnx")

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    onnx.helper.printable_graph(model.graph)