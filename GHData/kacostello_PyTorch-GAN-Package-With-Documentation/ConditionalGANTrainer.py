import SimpleGANTrainer
import ToTrain
import torch
import torch.nn.functional as func
import math
import numpy as np
import matplotlib.pyplot as plt


class ConditionalGANTrainer(SimpleGANTrainer.SimpleGANTrainer):
    def __init__(self, generator, discriminator, latent_space_function, random_from_dataset, g_loss, d_loss, g_opt,
                 d_opt, device, tt=None, d_thresh=0.5, num_input_variables=1, classes=1, do_wass_viz=True):
        """Class to train a Conditional GAN.
        Generator and discriminator are torch model objects
        Latent_space_function(n) is a function which returns an array of n points from the latent space
        Random_from_dataset is a function which returns an array of n points from the real dataset
        device is the pytorch device which models should be on.
        d_thresh is an optional parameter used to determine the threshold for a positive result from the discriminator.
        Used for visualizations."""
        SimpleGANTrainer.SimpleGANTrainer.__init__(self, generator, discriminator, latent_space_function,
                                                   random_from_dataset, g_loss, d_loss, g_opt, d_opt, device, tt,
                                                   d_thresh)

        self.classes = classes
        self.num_input_variables = num_input_variables
        self.do_wass_viz = do_wass_viz

        # Make array to hold wass dist information by epoch
        self.stats["W_Dist"] = {}
        for class_num in range(self.classes):
            self.stats["W_Dist"][class_num] = []

    def do_viz(self, tt, y, mod_loss, mod_pred, w_dist_mean):
        self.do_simple_viz(tt, y, mod_loss, mod_pred, w_dist_mean)
        self.wass_viz()

    def wass_viz(self):
        if self.do_wass_viz:
            for class_num in range(self.classes):
                # Obtain batch of fake data
                lat_space_data = self.generate_fake(class_num, self.num_input_variables, 500)
                fake_batch = self.eval_generator(lat_space_data)
                data_col = torch.arange(0, fake_batch.shape[1] - self.classes).to(self.device)
                striped_fake_batch = torch.index_select(fake_batch, 1, data_col)
                # Obtain batch of real data
                real_batch = self.dataset(500, self.device, class_num)
                striped_real_batch = torch.index_select(real_batch, 1, data_col)
                # Find and record the Wasserstein Distance
                was_dist = self.all_Wasserstein_dists(striped_fake_batch, striped_real_batch).mean()
                self.stats["W_Dist"][class_num].append(was_dist.item())

    def generate_fake(self, labelNum, num_input_variables, batch_size):
        data = torch.rand(batch_size, num_input_variables, device=self.device)
        labels = torch.from_numpy(np.ones(batch_size).astype(int) * labelNum).to(self.device)
        labels = func.one_hot(labels.to(torch.int64), num_classes=self.classes)
        labels = labels.reshape(batch_size, self.classes)
        return torch.cat((data, labels), 1)

    def divergence_by_epoch(self):
        ax = plt.subplot(111)
        for the_class in range(self.classes):
            plt.plot(list(range(1, len(self.stats["W_Dist"][the_class]) + 1)),
                     self.stats["W_Dist"][the_class], label="Class: " + str(the_class))
        plt.xlabel("Epochs")
        plt.ylabel("Wasserstein Distance")
        plt.title("Wasserstein Distance by Epochs")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
