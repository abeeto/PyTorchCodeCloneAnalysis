import torch
from model_utils import create_conv_encoder

kernel = 3
padding = 1
stride = 1


class CNN_classifier(torch.nn.Module):
    def __init__(self, device, args, image_shape, num_classes):
        super().__init__()
        self.layers = []
        try:
            dropout_rate = args.dropout_rate
        except AttributeError:
            dropout_rate = 0
        model_size = args.model_size
        if model_size not in ["big", "small"]:
            print(f"Unknown {model_size} size for the model. Please choose from the list:[big, small]")
            print("Exiting...")
            exit(1)
        # self.hidden_layers = [64, 128, 256, 512, 512]
        if model_size == "big":
            # self.hidden_layers = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]        # VGG11
            self.hidden_layers = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512]        # VGG9

        else:
            self.hidden_layers = [64, 'M', 128, 'M', 256, 256]
        in_channels = image_shape[0]
        # in_channels = 3
        adaptive_pooling_features = 7
        linear_dim_in = adaptive_pooling_features * adaptive_pooling_features * self.hidden_layers[-1]
        linear_dim_out = 8 * self.hidden_layers[-1]
        self.encoder_cnn, self.feature_map_size = create_conv_encoder(in_channels=in_channels,
                                                                      hidden_layers=self.hidden_layers,
                                                                      device=device)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((adaptive_pooling_features, adaptive_pooling_features))
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(linear_dim_in, linear_dim_out),
            torch.nn.ReLU(True),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(linear_dim_out, linear_dim_out),
            torch.nn.ReLU(True),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(linear_dim_out, num_classes),
        )

    def forward(self, x):
        conv_x = self.encoder_cnn(x)
        adaptive_mpooling = self.avgpool(conv_x)
        flatten_x = self.flatten(adaptive_mpooling)
        logits = self.classifier(flatten_x)
        return logits
