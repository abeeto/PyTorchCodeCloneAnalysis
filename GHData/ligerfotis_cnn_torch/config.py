kernel_size = 3
save_integral = 10
# dataset = "MNIST"
# dataset = "CIFAR10"
# dataset = "ImageNet"
conv_encoder_hidden_layers = [16, 32, 64, 128, 256]
conv_decoder_hidden_layers = [128, 64, 32, 16, 8]
#
# if dataset in ["MNIST", "mnist", "Mnist"]:
#     input_height = 28
#     input_width = 28
#     in_channel_size = 1
#
# elif dataset in ["CIFAR10", "cifar10", "Cifar10", "CIFAR-10"]:
#     input_height = 32
#     input_width = 32
#     in_channel_size = 3