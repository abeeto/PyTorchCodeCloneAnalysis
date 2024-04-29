from torch import nn
import torch


def init_weights_gaussian(m, std = 0.02):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d)\
         or isinstance(m, nn.Embedding) or isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, 0, std)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
    def forward(self, input):
        return input.view(input.size(0), -1)


class Discriminator(nn.Module):
    def __init__(self, n_classes = 10):
        super(Discriminator, self).__init__()
        self.n_classes = n_classes
        self.model = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,stride = 2, bias = False),
                                   nn.LeakyReLU(0.2, True),
                                   nn.Dropout(0.5),
                                   nn.Conv2d(in_channels=16, out_channels=16*2, kernel_size=3,stride = 1, bias = False),
                                   nn.BatchNorm2d(16*2),
                                   nn.LeakyReLU(0.2,True),
                                   nn.Dropout(0.5),
                                   nn.Conv2d(in_channels=16*2, out_channels=16 * 4, kernel_size=3, stride=2, bias=False),
                                   nn.BatchNorm2d(16 * 4),
                                   nn.LeakyReLU(0.2, True),
                                   nn.Dropout(0.5),
                                   nn.Conv2d(in_channels=16*4, out_channels=16 * 8, kernel_size=3, stride=1, bias=False),
                                   nn.BatchNorm2d(16 * 8),
                                   nn.LeakyReLU(0.2, True),
                                   nn.Dropout(0.5),
                                   nn.Conv2d(in_channels=16*8, out_channels=16 * 16, kernel_size=3, stride=2, bias=False),
                                   nn.BatchNorm2d(16 * 16),
                                   nn.LeakyReLU(0.2, True),
                                   nn.Dropout(0.5),
                                   nn.Conv2d(in_channels=16*16, out_channels=16 * 16 * 2, kernel_size=3, stride=1, bias=False),
                                   nn.BatchNorm2d(16 * 16 * 2),
                                   nn.LeakyReLU(0.2, True),
                                   nn.Dropout(0.5),
                                   Flatten())
        self.model.apply(init_weights_gaussian)
        # 61952 is for a 128*128 RGB input image
        self.output_discriminator = nn.Sequential(nn.Linear(61952,1), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Linear(61952, self.n_classes), nn.Softmax(dim=1))

    def forward(self, inputs):
        last_layer = self.model(inputs)
        return self.output_discriminator(last_layer), self.classifier(last_layer)

class Generator(nn.Module):
    def __init__(self, latent_vec_dim = 100, n_classes = 10, embedding_dim = 50) -> None:
        super(Generator,self).__init__()
        self.feature_map_dim = 8
        self.no_of_features = 768
        # to the embedding layer the label is fed (shape -> (1)) and the output can be reshaped to (1,32,32)
        # I might play around with this 
        self.label_embedding_nn = nn.Sequential(nn.Embedding(n_classes, embedding_dim),nn.Linear(embedding_dim, self.feature_map_dim ** 2))
        
        self.label_embedding_nn.apply(init_weights_gaussian)
        # the latent vector is initially fed to this network
        self.latent_vector_initial_nn = nn.Sequential(nn.Linear(latent_vec_dim, (self.no_of_features-1)  * (self.feature_map_dim ** 2), nn.ReLU(True)))
        self.latent_vector_initial_nn.apply(init_weights_gaussian)
        # TODO
        # I should look a little bit into these and make an informed decision
        batch_norm_eps = 0.8
        batch_norm_momentum = 0.1
        kernel_size = 5        
        # 768 channels -> 384
        self.first_conv_layer = nn.Sequential(
                                nn.ConvTranspose2d(self.no_of_features, int(self.no_of_features/2), kernel_size, stride = 2, padding=2, output_padding=1),
                                nn.BatchNorm2d(int(self.no_of_features/2), eps = batch_norm_eps, momentum=batch_norm_momentum),
                                nn.ReLU(True))
        # 384 -> 256
        self.second_conv_layer = nn.Sequential(
                                nn.ConvTranspose2d(int(self.no_of_features/2), int(self.no_of_features/3), kernel_size, stride = 2, padding=2, output_padding=1),
                                nn.BatchNorm2d(int(self.no_of_features/3), eps = batch_norm_eps, momentum=batch_norm_momentum),
                                nn.ReLU(True))
        # 256 -> 192
        self.third_conv_layer = nn.Sequential(nn.ConvTranspose2d(int(self.no_of_features/3), int(self.no_of_features/4), kernel_size, stride = 2, padding = 2, output_padding=1),
                                nn.BatchNorm2d(int(self.no_of_features/4), eps = batch_norm_eps, momentum=batch_norm_momentum),
                                nn.ReLU(True))
        # 192 -> 3
        self.fourth_conv_layer = nn.Sequential(nn.ConvTranspose2d(int(self.no_of_features/4), 3, kernel_size, stride=2, padding=2, output_padding=1),
                                nn.Tanh())

        self.conv_layers = nn.Sequential(self.first_conv_layer, self.second_conv_layer, self.third_conv_layer, self.fourth_conv_layer) 
        self.conv_layers.apply(init_weights_gaussian)
    def forward(self, inputs):
        latent_vector, label = inputs
        embedded_label = self.label_embedding_nn(label)
        embedded_label = embedded_label.view(-1,1,self.feature_map_dim, self.feature_map_dim)
        latent_vector_as_feature_maps = self.latent_vector_initial_nn(latent_vector)
        latent_vector_as_feature_maps = latent_vector_as_feature_maps.view(-1,self.no_of_features-1, self.feature_map_dim, self.feature_map_dim)
        concatenated_features_maps = torch.cat((latent_vector_as_feature_maps, embedded_label), 1)
        img = self.conv_layers(concatenated_features_maps) 
        return img


binary_crossentropy_loss = nn.BCELoss()
categorical_crosentropy_loss = nn.NLLLoss()

# I return both the loss of the discriminative part and the loss of the classifying part
def eval_loss(origin_prediction, actual_origin, label_prediction, real_label):
    return binary_crossentropy_loss(origin_prediction, actual_origin),\
        categorical_crosentropy_loss(label_prediction, real_label)


def save_checkpoint(PATH, epoch, generator, discriminator, gen_optimizer, discriminator_optimizer, g_loss, d_loss):
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'gen_optimizer_state_dict': gen_optimizer.state_dict(),
        'g_loss': g_loss,
        'discriminator_state_dict': discriminator.state_dict(),
        'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
        'd_loss': d_loss,
    }, PATH)