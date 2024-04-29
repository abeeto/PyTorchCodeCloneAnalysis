model_pytorch = MonoDataset(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
model_pytorch.load_state_dict(torch.load('./models/model_simple.pt'))

# Single pass of dummy variable required
dummy_input = torch.from_numpy(X_test[0].reshape(1, -1)).float().to(device)
dummy_output = model_pytorch(dummy_input)
print(dummy_output)

# Export to ONNX format
torch.onnx.export(model_pytorch, dummy_input, './models/mono_dataset.onnx', input_names=['input'], output_names=['output'])


