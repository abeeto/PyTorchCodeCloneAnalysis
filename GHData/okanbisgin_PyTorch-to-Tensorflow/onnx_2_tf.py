model_onnx = onnx.load('./models/mono_dataset.onnx')

tf_rep = prepare(model_onnx)

# Print out tensors and placeholders in model (helpful during inference in TensorFlow)
print(tf_rep.tensor_dict)

# Export model as .pb file
tf_rep.export_graph('./models/mono_dataset.pb')