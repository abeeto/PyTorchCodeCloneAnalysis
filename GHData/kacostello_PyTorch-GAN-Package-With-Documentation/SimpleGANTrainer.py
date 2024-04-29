import ToTrain
import torch
import torch.nn.functional as func
import math, os, pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import torch.optim as optim



class SimpleGANTrainer:
    def __init__(self, generator, discriminator, latent_space_function, random_from_dataset, g_loss, d_loss, g_opt,
                 d_opt, device, tt=None, d_thresh=0.5):
        """Class to train a simple GAN.
        Generator and discriminator are torch model objects
        Latent_space_function(n) is a function which returns an array of n points from the latent space
        Random_from_dataset is a function which returns an array of n points from the real dataset
        device is the pytorch device which models should be on.
        d_thresh is an optional parameter used to determine the threshold for a positive result from the discriminator.
        Used for visualizations."""
        if tt is None:
            self.totrain = ToTrain.TwoFiveRule()
        else:
            self.totrain = tt
        self.dataset = random_from_dataset
        self.latent_space = latent_space_function

        self.models = {"G": generator.to(device), "D": discriminator.to(device)}
        self.in_functions = {"G": self.generator_input, "D": self.discriminator_input}
        self.loss_functions = {"G": g_loss, "D": d_loss}
        self.optimizers = {"G": g_opt, "D": d_opt}
        self.stats = {}  # Dictionary to keep track of the stats we want to save. Of the format {stat_name:stat_dict}

        self.stats["losses"] = {"G": [], "D": []}
        self.stats["epochs_trained"] = {"G": 0, "D": 0}
        self.stats["d_fpr"] = []
        self.stats["d_recall"] = []
        self.stats["d_precision"] = []

        self.d_thresh = d_thresh
        self.device = device
        self.flags = {"is_wass":False}

    def train(self, n_epochs, n_batch):
        for epoch in range(n_epochs):
            tt = self.totrain.next(self)  # Determine which model to train - sw will either be "D" or "G"

            # Both input functions return the tuple (dis_in, labels)
            # generator_in returns (gen_out, labels) - this data is passed through D and used to train G
            # discriminator_in returns (dis_in, labels) - this is used to train D directly
            # For other GAN types: input functions can return whatever makes the most sense for your specific GAN
            # (so controllable GAN, for instance, might want to return a classification vector as well)
            dis_in, y = self.in_functions[tt](n_batch)
            if tt == "G":  # If we're training the generator, we should temporarily put the discriminator in eval mode
                self.models["D"].eval()
            mod_pred = self.models["D"](dis_in)
            self.models["D"].train()
            mod_loss = self.loss_functions[tt](mod_pred, y)

            # Pytorch training steps
            self.optimizers[tt].zero_grad()
            mod_loss.backward()
            self.optimizers[tt].step()

            if self.flags["is_wass"]:
                if tt == "D":
                    for p in self.models["D"].parameters():
                        p.data.clamp_(-0.01, 0.01)

                w_dists = self.all_Wasserstein_dists(self.eval_generator(self.latent_space(256)), self.dataset(256))
                w_dist_mean = torch.mean(w_dists)

                self.do_viz(tt, y, mod_loss, mod_pred, w_dist_mean)
            else:
                self.do_viz(tt, y, mod_loss, mod_pred, None)

    def to_wass(self, g_lr, d_lr):
        g_opt = optim.RMSprop(self.models["G"].parameters(), g_lr)
        d_opt = optim.RMSprop(self.models["D"].parameters(), d_lr)
        self.optimizers = {"G": g_opt, "D": d_opt}
        self.flags["is_wass"] = True
        self.stats["wass_dists"] = []

    def do_viz(self, tt, y, mod_loss, mod_pred, w_dist_mean):
        self.do_simple_viz(tt, y, mod_loss, mod_pred, w_dist_mean)

    def do_simple_viz(self, tt, y, mod_loss, mod_pred, w_dist_mean):
        # Logging for visualizers
        self.stats["losses"][tt].append(mod_loss.item())
        self.stats["epochs_trained"][tt] += 1

        y_flat = y.cpu().numpy().flatten()  # Calculate fPr, recall, precision
        mod_pred_flat = mod_pred.cpu().detach().numpy().flatten()
        fP = 0
        fN = 0
        tP = 0
        tN = 0
        for i in range(len(y_flat)):
            if y_flat[i] == 0:
                if mod_pred_flat[i] > self.d_thresh:
                    fP += 1
                else:
                    tN += 1
            else:
                if mod_pred_flat[i] > self.d_thresh:
                    tP += 1
                else:
                    fN += 1

        if fP + tN > 0:
            self.stats["d_fpr"].append(fP / (fP + tN))
        if tP + fP > 0:
            self.stats["d_precision"].append(tP / (tP + fP))
        if tP + fN > 0:
            self.stats["d_recall"].append(tP / (tP + fN))

        if w_dist_mean is not None:
            self.stats["wass_dists"].append(w_dist_mean)

    def eval_generator(self, in_dat):
        return self.eval("G", in_dat)

    def eval_discriminator(self, in_dat):
        return self.eval("D", in_dat)

    def get_g_loss_fn(self):
        return self.loss_functions["G"]

    def get_g_opt_fn(self):
        return self.optimizers["G"]

    def get_d_loss_fn(self):
        return self.loss_functions["D"]

    def get_d_opt_fn(self):
        return self.optimizers["D"]

    def loss_by_epoch_g(self):
        self.loss_by_epoch("G")

    def loss_by_epoch_d(self):
        self.loss_by_epoch("D")

    def discriminator_input(self, n_batch):
        gen_in = self.latent_space(math.ceil(n_batch / 2), self.device)
        self.models["G"].eval()
        gen_out = self.models["G"](gen_in)
        self.models["G"].train()
        dis_in = torch.cat((gen_out, self.dataset(int(n_batch / 2), self.device)))
        y = torch.tensor([[0] for _ in range(math.ceil(n_batch / 2))] + [[1] for _ in range(int(n_batch / 2))],
                         device=self.device).float()  # TODO: used .float() here because the model I'm using to test
        # uses floats. Find a way to automatically find the correct data type
        return dis_in, y

    def generator_input(self, n_batch):
        gen_in = self.latent_space(n_batch, self.device)
        gen_out = self.models["G"](gen_in)
        y = torch.tensor([[1] for _ in range(n_batch)], device=self.device).float()
        return gen_out, y

    def __eq__(self, other):
        # TODO: This doesn't actually guarantee that each model is strictly *equal*, but it's good enough for checking if save/load works

        if self.stats != other.stats:
            return False
        for i in self.in_functions:
            if i not in other.in_functions or other.in_functions[i] is None:
                return False
        for i in self.loss_functions:
            if i not in other.loss_functions or other.loss_functions[i] is None:
                return False
        if self.list_opts() != other.list_opts():
            return False
        if self.list_models() != other.list_models():
            return False
        return True

    def eval(self, model, in_dat):
        self.models[model].eval()
        out = self.models[model](in_dat.to(device=self.device))
        self.models[model].train()
        return out

    def loss_by_epoch(self, model):  # TODO: format the graph nicely
        model_loss = self.stats["losses"][model]
        plt.plot(model_loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss by Epochs")
        plt.show()

    def epochs_trained(self, model):
        return self.stats["epochs_trained"][model]

    def total_epochs_trained(self):
        total = 0
        for model in self.list_models():
            total += self.epochs_trained(model)
        return total

    def list_models(self):
        return [n for n in
                self.models]  # Kinda scuffed code to get a list of a dict's keys. do not remember the actual way

    def list_opts(self):
        return [n for n in self.optimizers]

    def save_flags(self, path):
        """Saves the trainer object's flags to folder at specified location. Creates folder if it does not exist.
        Saves flags as a pickle, in format path/trainer_flags.ts. Not (yet) guaranteed to be compatible across different
        versions!
        WARNING: Will overwrite the existing trainer_flags file, if it exists."""
        if not os.path.isdir(path):
            os.mkdir(path)

        f = open(path + "\\trainer_flags.ts", "wb")
        pickle.dump(self.flags, f)
        f.close()

    def load_and_check_flags(self, path):
        """Loads trainer flags dict from specified folder. Throws ValueError if path does not exist, or if
        path/trainer_flags.ts does not exist. Requires the trainer object to already be instantiated. Will
        overwrite the current trainer flags dict. Also, checks flags for compatibility.
        Throws ValueError if the save's flags are incompatible with the trainer's current flags
        WARNING: This load function uses pickle. *Only attempt to load from a trusted source.*"""
        try:
            assert os.path.isdir(path)
        except AssertionError:
            raise ValueError("No folder at " + path)

        try:
            assert os.path.isfile(path + "\\trainer_flags.ts")
        except AssertionError:
            raise ValueError("Cannot detect trainer flags dict at " + path)

        f = open(path + "\\trainer_flags.ts", "rb")
        newflags = pickle.load(f)
        # Check for compat
        if self.flags["is_wass"] != newflags["is_wass"]:
            if self.flags["is_wass"]:
                raise ValueError("Cannot load non-wasserstein checkpoint as a wasserstein trainer!")
            else:
                raise ValueError("Cannot load wasserstein checkpoint as a non-wasserstein trainer!")
        self.flags = newflags
        f.close()

    def save_model_state_dicts(self, path):
        """Saves the model state dicts to folder at specified location. Creates folder if it does not exist.
        Saves state dicts in format path/model_name.pt for each model_name in self.models.
        WARNING: Will overwrite existing state_dicts, if present."""
        if not os.path.isdir(path):
            os.mkdir(path)

        for model_name in self.list_models():
            torch.save(self.models[model_name].state_dict(), path + "\\" + model_name + ".pt")

    def save_opt_state_dicts(self, path):
        """Saves the optimizer state dicts to folder at specified location. Creates folder if it does not exist.
        Saves state dicts in format path/optimizer_name.pto for each optimizer_name in self.optimizers.
        WARNING: Will overwrite existing state_dicts, if present."""
        if not os.path.isdir(path):
            os.mkdir(path)

        for opt_name in self.list_opts():
            torch.save(self.optimizers[opt_name].state_dict(), path + "\\" + opt_name + ".pto")

    def load_model_state_dicts(self, path):
        """Loads model state dicts from specified folder. Throws ValueError if path does not exist,
        or does not have *all* necessary state dicts. Requires models to already be instantiated with the
        same structure as saved!
        WARNING: the torch.load() function uses pickle. *Only attempt to load state_dicts from a trusted source.*"""
        try:
            assert os.path.isdir(path)
        except AssertionError:
            raise ValueError("No folder at " + path)

        try:
            for model_name in self.list_models():
                assert os.path.isfile(path + "\\" + model_name + ".pt")
        except AssertionError:
            raise ValueError("Not all models have an associated state_dict at " + path)

        for model_name in self.list_models():
            self.models[model_name].load_state_dict(torch.load(path + "\\" + model_name + ".pt"))

    def load_opt_state_dicts(self, path):
        """Loads optimizer state dicts from specified folder. Throws ValueError if path does not exist,
        or does not have *all* necessary state dicts. Requires optimizers to already be instantiated with the
        same structure as saved!
        WARNING: the torch.load() function uses pickle. *Only attempt to load state_dicts from a trusted source.*"""
        try:
            assert os.path.isdir(path)
        except AssertionError:
            raise ValueError("No folder at " + path)

        try:
            for opt_name in self.list_opts():
                assert os.path.isfile(path + "\\" + opt_name + ".pto")
        except AssertionError:
            raise ValueError("Not all optimizers have an associated state_dict at " + path)

        for opt_name in self.list_opts():
            self.optimizers[opt_name].load_state_dict(torch.load(path + "\\" + opt_name + ".pto"))

    def save_trainer_stats_dict(self, path):
        """Saves trainer stats dict into specified folder. Creates folder if it does not exist.
        Saves state dict in format path/trainer_stats.ts. Not (yet) guaranteed to be compatible across different
        versions!
        WARNING: Will overwrite the existing trainer_stats file, if it exists."""
        if not os.path.isdir(path):
            os.mkdir(path)

        f = open(path + "\\trainer_stats.ts", "wb")
        pickle.dump(self.stats, f)
        f.close()

    def save_loss_functions(self, path):
        """Saves loss functions dict into specified folder. Creates folder if it does not exist.
        Saves state dict in format path/loss_functions.ts.
        WARNING: Will overwrite the existing loss_functions file, if it exists."""
        if not os.path.isdir(path):
            os.mkdir(path)

        f = open(path + "\\loss_functions.ts", "wb")
        pickle.dump(self.loss_functions, f)
        f.close()

    def save_in_functions(self, path):
        """Saves in functions dict into specified folder. Creates folder if it does not exist.
        Saves state dict in format path/in_functions.ts.
        WARNING: Will overwrite the existing in_functions file, if it exists."""
        if not os.path.isdir(path):
            os.mkdir(path)

        f = open(path + "\\in_functions.ts", "wb")
        pickle.dump(self.in_functions, f)
        f.close()

    def save_to_train(self, path):
        """Saves to_train object into specified folder. Creates folder if it does not exist.
        Saves state dict in format path/to_train.ts.
        WARNING: Will overwrite the existing to_train file, if it exists."""
        if not os.path.isdir(path):
            os.mkdir(path)

        f = open(path + "\\to_train.ts", "wb")
        pickle.dump(self.totrain, f)
        f.close()

    def load_trainer_state_dict(self, path):
        """Loads trainer state dict from specified folder. Throws ValueError if path does not exist, or if
        path/trainer_stats.ts does not exist. Requires the trainer object to already be instantiated. Will
        overwrite the current trainer stats dict.
        WARNING: This load function uses pickle. *Only attempt to load from a trusted source.*"""
        try:
            assert os.path.isdir(path)
        except AssertionError:
            raise ValueError("No folder at " + path)

        try:
            assert os.path.isfile(path + "\\trainer_stats.ts")
        except AssertionError:
            raise ValueError("Cannot detect trainer stats dict at " + path)

        f = open(path + "\\trainer_stats.ts", "rb")
        self.stats = pickle.load(f)
        f.close()

    def load_loss_functions(self, path):
        """Loads loss_functions dict from specified folder. Throws ValueError if path does not exist, or if
        path/loss_functions.ts does not exist. Requires the trainer object to already be instantiated. Will
        overwrite the current loss_functions dict. Note that this requires the loss functions to be defined
        in the current scope, or otherwise defined within a built-in module (if you're using a custom loss
        function, make sure to import the file where it's defined!). If you're using pytorch's built-in loss functions,
        and you have pytorch properly installed, this should work without importing pytorch.
        WARNING: This load function uses pickle. *Only attempt to load from a trusted source.*"""
        try:
            assert os.path.isdir(path)
        except AssertionError:
            raise ValueError("No folder at " + path)

        try:
            assert os.path.isfile(path + "\\loss_functions.ts")
        except AssertionError:
            raise ValueError("Cannot detect loss_functions dict at " + path)

        f = open(path + "\\loss_functions.ts", "rb")
        self.loss_functions = pickle.load(f)
        f.close()

    def load_in_functions(self, path):
        """Loads in_functions dict from specified folder. Throws ValueError if path does not exist, or if
        path/in_functions.ts does not exist. Requires the trainer object to already be instantiated. Will
        overwrite the current in_functions dict. Note that this requires the in functions to be defined
        in the current scope, or otherwise defined within a built-in module (if you're using a custom
        function, make sure to import the file where it's defined!).
        WARNING: This load function uses pickle. *Only attempt to load from a trusted source.*"""
        try:
            assert os.path.isdir(path)
        except AssertionError:
            raise ValueError("No folder at " + path)

        try:
            assert os.path.isfile(path + "\\in_functions.ts")
        except AssertionError:
            raise ValueError("Cannot detect in_functions dict at " + path)

        f = open(path + "\\in_functions.ts", "rb")
        self.in_functions = pickle.load(f)
        f.close()

    def load_to_train(self, path):
        """Loads totrain object from specified folder. Throws ValueError if path does not exist, or if
        path/to_train.ts does not exist. Requires the trainer object to already be instantiated. Will
        overwrite the current totrain object. Note that this requires the totrain object to be defined
        in the current scope, or otherwise defined within a built-in module (if you're using a custom
        object, make sure to import the file where it's defined!).
        WARNING: This load function uses pickle. *Only attempt to load from a trusted source.*"""
        try:
            assert os.path.isdir(path)
        except AssertionError:
            raise ValueError("No folder at " + path)

        try:
            assert os.path.isfile(path + "\\to_train.ts")
        except AssertionError:
            raise ValueError("Cannot detect in_functions dict at " + path)

        f = open(path + "\\to_train.ts", "rb")
        self.totrain = pickle.load(f)
        f.close()

    def soft_save(self, path):
        """Saves all model state_dicts, the trainer's stat dict and flag dict, all optimizer state_dicts,
        all in_functions, and all loss_functions.
        Will overwrite previously saved dicts in same location!"""
        self.save_flags(path)
        self.save_model_state_dicts(path)
        self.save_trainer_stats_dict(path)
        self.save_loss_functions(path)
        self.save_in_functions(path)
        self.save_opt_state_dicts(path)
        self.save_to_train(path)

    def soft_load(self, path):
        """Loads all model state_dicts and the trainer's stat dict and flag dict.
        Note that this requires the loss functions and in functions to be defined in the current scope, or otherwise
        defined within a built-in module (if you're using a custom loss function, make sure to import the file where
        it's defined!). If you're using pytorch's built-in loss functions, and you have pytorch properly installed, you
        shouldn't need to import pytorch.
        WARNING: This load function uses pickle. *Only attempt to load from a trusted source.*"""
        self.load_and_check_flags(path)
        self.load_model_state_dicts(path)
        self.load_trainer_state_dict(path)
        self.load_in_functions(path)
        self.load_loss_functions(path)
        self.load_opt_state_dicts(path)
        self.load_to_train(path)

    def models_to(self, newdevice):
        self.device = newdevice
        for model in self.list_models():
            self.models[model].to(self.device)

    def all_Wasserstein_dists(self, fake, real):
        feature_dim = len(real[0])
        real = real.cpu().detach().numpy()
        fake = fake.cpu().detach().numpy()
        return torch.tensor([wasserstein_distance(fake[:, k], real[:, k]) for k in range(feature_dim)])
