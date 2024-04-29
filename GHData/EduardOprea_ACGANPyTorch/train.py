import sys
import torch
from torch import optim
from torch import nn
import torchvision
from acgan import Discriminator, Generator, eval_loss, save_checkpoint
from dataset import load_data
from torch.autograd import Variable
from argparse import ArgumentParser, Namespace
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def get_args() -> Namespace:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-d", "--directory",
                            type=str,
                            default="D:\Facultate\Cercetare\Code\CGAN example\clothing_items",
                            dest="dataset_path",
                            help="The path to the root of the dataset")
    arg_parser.add_argument("-e", "--epochs",
                            type=int,
                            default=100,
                            dest="nepochs",
                            help="The number of epochs for training")
    arg_parser.add_argument("-b", "--batchsize",
                            type=int,
                            default=100,
                            dest="batchsize",
                            help="The batch size at each iteration of training")
    arg_parser.add_argument("-lvd", "--latentvectordimension",
                            type=int,
                            default=100,
                            dest="latent_dim",
                            help="The size of the latent vector")
    arg_parser.add_argument("--lrdiscriminator",
                            type=float,
                            default=.0004,
                            dest="lr_d",
                            help="Learning rate for the discriminator optimizer")
    arg_parser.add_argument("--lrgenerator",
                            type=float,
                            default=.0001,
                            dest="lr_g",
                            help="Learning rate for the generator optimizer")                        
    arg_parser.add_argument("--b1",
                            type=float,
                            default=.5,
                            dest="b1",
                            help="Beta 1 of adam optimizer")
    arg_parser.add_argument("--b2",
                            type=float,
                            default=.999,
                            dest="b2",
                            help="Beta 2 of adam optimizer")

    return arg_parser.parse_args()

def generate_image_and_save(generator, epoch):
    noise_vector = torch.randn(1, args.latent_dim, device=device)
    noise_vector = noise_vector.to(device)
    labels = torch.randint(0,13,(1,1)).to(device)
    print(labels)
    generated_image = generator((noise_vector, labels))[0]
    print(generated_image.shape)
    generated_image = generated_image.view(generated_image.shape[1], generated_image.shape[2], generated_image.shape[0])
    print(generated_image.shape)
    plt.imshow(generated_image.detach().cpu())
    plt.savefig(F"generated_samples/fake_sample_epoch_{epoch}")
    plt.close()

if __name__ == '__main__':
    args = get_args()
    
    device = 'cpu'
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Working on {device}")
    dataloader = load_data(args.dataset_path, args.batchsize)
    

    for index, (real_images, labels) in enumerate(dataloader):
        for idx, image in enumerate(real_images):
            reshaped_img = torch.transpose(image, 0, 2)
            reshaped_img = torch.transpose(reshaped_img, 0, 1)
            #reshaped_img = image.view(image.shape[1], image.shape[2], image.shape[0])
            plt.imshow(reshaped_img)
            plt.savefig(f'test_dataset/img{index}_{idx}.png')
            plt.close()

    writer = SummaryWriter("results/acgan")
    examples = iter(dataloader)
    example_data, example_targets = examples.next()
    test_img = example_data[0].view(example_data[0].shape[1], example_data[0].shape[2], example_data[0].shape[0])
    plt.imshow(test_img)
    plt.show()
    # for i in range(6):
    #     plt.subplot(2,3,i+1)
    #     plt.imshow(example_data[i][0])
    # #plt.show()
    # img_grid = torchvision.utils.make_grid(example_data)
    # writer.add_image('fashion_dataset', img_grid)
    
    print(f'Dataset loaded, number of classes => {len(dataloader.dataset.classes)}')
    generator = Generator(latent_vec_dim=args.latent_dim, n_classes=len(dataloader.dataset.classes)).to(device)
    discriminator = Discriminator(n_classes=len(dataloader.dataset.classes)).to(device)
    G_optimizer = optim.Adam(generator.parameters(), lr=args.lr_g, betas=(args.b1, args.b2))
    D_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(args.b1, args.b2))
    # writer.add_graph(discriminator, example_data)
    writer.close()
    epochs_to_save = 5
    nsteps_epoch = len(dataloader)
    for epoch in range(args.nepochs):
        print(f"Current epoch: {epoch}")
        D_loss_list, G_loss_list = [], []

        for index, (real_images, labels) in enumerate(dataloader):
            print(f"Current step for current epoch: {index}/{nsteps_epoch}")
            
            D_optimizer.zero_grad()
            real_images = real_images.to(device)
            labels = labels.to(device)
            labels_unsqueezed = labels
            labels = labels.unsqueeze(1).long()

            real_target = torch.ones(real_images.size(0), 1, requires_grad=True).to(device)
            fake_target = torch.zeros(real_images.size(0), 1, requires_grad=True).to(device)

            origin_prediction, label_prediction = discriminator(real_images)
            # evaluate discriminator loss for a batch of real samples
            discriminator_loss_real, classifier_loss_real =\
                eval_loss(origin_prediction, real_target, label_prediction, labels_unsqueezed)


            D_loss_real = discriminator_loss_real + classifier_loss_real
            D_loss_real.backward()
            D_optimizer.step()
            noise_vector = torch.randn(real_images.size(0), args.latent_dim, device=device)
            noise_vector = noise_vector.to(device)


            generated_images = generator((noise_vector, labels_unsqueezed))
            origin_prediction_fake, label_prediction_fake = discriminator(generated_images.detach())

            # evaluate discriminator loss for a batch of fake samples produced by the generator
            discriminator_loss_fake, classifier_loss_fake =\
                eval_loss(origin_prediction_fake, fake_target, label_prediction_fake, labels_unsqueezed)

            D_loss_fake = discriminator_loss_fake + classifier_loss_fake
            D_loss_fake.backward()
            D_optimizer.step()

            D_total_loss = discriminator_loss_fake + discriminator_loss_fake +\
                classifier_loss_fake + classifier_loss_real
            D_loss_list.append(D_total_loss)
            if discriminator_loss_fake < 0 or discriminator_loss_real < 0 or\
                 classifier_loss_fake < 0 or classifier_loss_real < 0 :
                  print("Stop here")

            print(f"Discriminator loss fake sample -> {discriminator_loss_fake},\
                 Discriminator loss real sample -> {discriminator_loss_real}\
                  Classifier loss fake sample -> {classifier_loss_fake}\
                      Classifier loss real sample -> {classifier_loss_real}")
            G_optimizer.zero_grad()
            origin_prediction_fake, label_prediction_fake = discriminator(generated_images)
            generator_loss_discriminator, generator_loss_classifier =\
                 eval_loss(origin_prediction_fake, real_target, label_prediction_fake, labels_unsqueezed) 
            # the generator is trying to make the discriminator categorize the sample produced by him as real
            # and to make the classifier identify the labels
            G_total_loss = generator_loss_discriminator + generator_loss_classifier
            G_loss_list.append(G_total_loss)

            G_total_loss.backward()
            G_optimizer.step()

        if (epoch % epochs_to_save) == 0:
          model_intermediary = F"cgan_fashion_epoch{epoch}.pth.tar"
          intermediary_path = F"/content/gdrive/My Drive/cgan_fashion/training_models/{model_intermediary}"
          save_checkpoint(intermediary_path, epoch, generator, discriminator, G_optimizer, D_optimizer, G_loss_list, D_loss_list)
        # at the end of each epoch I generate a sample image for evaluation
        generate_image_and_save(generator, epoch)
