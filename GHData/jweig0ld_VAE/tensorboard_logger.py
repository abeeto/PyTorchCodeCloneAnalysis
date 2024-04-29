import torch
import tensorboard as tb
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F


def log_performance(writer, batch_size, start_time, end_time, step):
    examples_per_second = batch_size / (end_time - start_time)
    writer.add_scalar("Examples per second", examples_per_second, step)


def log_images_3d(model, images, test_batch, step):
    model.eval()
    _, probs, _, _ = model(images)
    for i in range(len(images)):
        block = images[i]
        block = block.permute((1, 0, 2, 3))
        recon_block = probs[i].permute((1, 0, 2, 3))
        original_images = torch.cat(tuple(block), 2)
        recon_images = torch.cat(tuple(recon_block), 2)
        model.writer.add_image('Training {}/Original'.format(i), original_images, step)
        model.writer.add_image('Training {}/Reconstructed'.format(i), recon_images, step)
    _, probs, _, _ = model(test_batch)
    for i in range(len(test_batch)):
        block = test_batch[i]
        block = block.permute((1, 0, 2, 3))
        recon_block = probs[i].permute((1, 0, 2, 3))
        original_images = torch.cat(tuple(block), 2)
        recon_images = torch.cat(tuple(recon_block), 2)
        model.writer.add_image('Testing {}/Original'.format(i), original_images, step)
        model.writer.add_image('Testing {}/Reconstructed'.format(i), recon_images, step)


def log_grads(model, step: int):
    """
    :param model: Conv2DVAE or Conv3dVAE object, defined in vae_skeleton.py.
    :param step: The number of gradient steps that have been performed during training.
    """
    model.train()
    n_layers = len(model.json_data['layers'])
    for i in range(2 * n_layers - 1):
        if i % 2 == 0:
            encoder_grads = model.encoder[i].weight.grad
            decoder_grads = model.decoder[i].weight.grad
            model.writer.add_histogram("Conv Layer {}/Gradients".format(i), encoder_grads, step)
            model.writer.add_histogram("Deconv Layer {}/Gradients".format(i), decoder_grads, step)
            model.writer.add_scalar("Gradient Norm/Conv Layer {}".format(i), torch.norm(encoder_grads), step)
            model.writer.add_scalar("Gradient Norm/Deconv Layer {}".format(i), torch.norm(decoder_grads), step)

    # Bottleneck layers
    mu_grads = model.mu_layer.weight.grad
    logvar_grads = model.logvar_layer.weight.grad
    transition_grads = model.transition_layer.weight.grad
    model.writer.add_histogram("Mu Layer/Gradients", mu_grads, step)
    model.writer.add_histogram("Logvar Layer/Gradients", logvar_grads, step)
    model.writer.add_histogram("Transition Layer/Gradients", transition_grads, step)
    model.writer.add_scalar("Gradient Norm/Mu Layer", torch.norm(mu_grads), step)
    model.writer.add_scalar("Gradient Norm/Logvar Layer", torch.norm(logvar_grads), step)
    model.writer.add_scalar("Gradient Norm/Transition Layer", torch.norm(transition_grads), step)


def log_activations(model, control_batch, step: int):
    """
    :param model: Conv2DVAE or Conv3dVAE object, defined in vae_skeleton.py.
    :param control_batch: A single batch of training/testing data.
    :param step: The number of gradient steps that have been performed during training.
    :return:
    """
    pass
