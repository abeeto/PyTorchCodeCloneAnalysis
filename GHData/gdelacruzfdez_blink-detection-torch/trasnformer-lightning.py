import os
import torch
from torch import nn, einsum
import torch.nn.functional as F
import pytorch_lightning as pl

import network
from vivit.module import Attention, PreNorm, FeedForward
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from PIL import Image
from torchvision.transforms import transforms
import datamodule
from sklearn import metrics
import numpy as np
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ray import tune
from ray.tune import CLIReporter
import evaluator
from numpy.lib.stride_tricks import sliding_window_view
from functools import partial
from augmentator import TransformerImgAugTransform
import torchvision.models as models

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class VivitTransformer(pl.LightningModule):
    def __init__(self, config, train_weights, datamodule, pool='cls'):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_classes = config["num_classes"]
        self.num_frames = config["num_frames"]
        self.dim = config["dim"]
        self.batch_size = config["batch_size"]
        self.depth = config["depth"]
        self.heads = config["heads"]
        self.in_channels = config["in_channels"]
        self.dim_head = config["dim_head"]
        self.dropout = config["dropout"]
        self.emb_dropout = config["emb_dropout"]
        self.scale_dim = config["scale_dim"]
        self.lr = config["lr"]

        self.datamodule = datamodule

        self.train_weights = torch.FloatTensor(train_weights).cuda()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        assert self.image_size % self.patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (self.image_size // self.patch_size) ** 2
        patch_dim = self.in_channels * self.patch_size ** 2

        self.efficient_net = models.efficientnet_b0(pretrained=True)
        self.efficient_net.classifier = nn.Identity()

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_frames, num_patches + 1, self.dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.space_transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.dim * self.scale_dim,
                                             self.dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.temporal_transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head,
                                                self.dim * self.scale_dim, self.dropout)

        self.dropout = nn.Dropout(self.emb_dropout)
        self.pool = pool

        self.cnn_model = [network.SiameseNetV2(self.dim)]
        self.cnn_model[0].load_state_dict(torch.load('/home/gcruz/hdd/paper_models/eval_params/blink_detection/models/cnn/model.pt'))


        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim ),
            nn.Linear(self.dim, self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        print(x.size(), self.pos_embedding[:, :, :(n + 1)].size())
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 1:]
        x = x.reshape(-1, self.dim)

        return self.mlp_head(x)

    def loss_function(self, x_hat, y):
        return F.cross_entropy(x_hat, y, weight=self.train_weights)


    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = self.__fill_features_if_needed(x, y)
        x = x.view(-1, self.num_frames, self.in_channels, self.image_size, self.image_size)
        x_hat = self(x)
        loss = self.loss_function(x_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = self.__fill_features_if_needed(x, y)
        x = x.view(-1, self.num_frames, self.in_channels, self.image_size, self.image_size)
        x_hat = self(x)
        loss = self.loss_function(x_hat, y)
        self.log("val_loss", loss)
        probabilities = x_hat[:, 1].data.cpu()
        _, predictions = torch.max(x_hat.data, 1)
        predictions = np.array(predictions.data.cpu()).reshape(-1)
        y = np.array(y.data.cpu()).reshape(-1)
        return batch_idx, y, predictions, loss.data.cpu(), probabilities

    def validation_epoch_end(self, outputs):
        batch_idx, y, predictions, losses, probabilities = list(zip(*outputs))
        y = np.concatenate(y)
        predictions = np.concatenate(predictions)
        probabilities = np.concatenate(probabilities)

        dataframe = self.datamodule.get_val_dataframe()
        predictions = predictions[:len(dataframe)]
        y = y[:len(dataframe)]
        probabilities = probabilities[:len(dataframe)]
        moving_avg_probabilities = self.moving_average(probabilities)
        moving_avg_pred = moving_avg_probabilities > 0.7
        moving_avg_big_probabilities = self.moving_average_big(probabilities)
        moving_avg_big_pred = moving_avg_big_probabilities > 0.7
        print(probabilities.size, moving_avg_probabilities.size)
        dataframe['pred'] = predictions
        dataframe['target'] = y
        dataframe['probabilities'] = probabilities
        dataframe['moving_avg_probabilities'] = moving_avg_probabilities
        dataframe['moving_avg_pred'] = moving_avg_pred
        dataframe['moving_avg_big_probabilities'] = moving_avg_big_probabilities
        dataframe['moving_avg_big_pred'] = moving_avg_big_pred
        #print(evaluator.BlinkDetectionEvaluator().evaluate(dataframe))
        dataframe.to_csv("transformer_results.csv")
        #dataframe['pred'] = moving_avg_pred
        #print(evaluator.BlinkDetectionEvaluator().evaluate(dataframe))
        dataframe['pred'] = moving_avg_big_pred
        #print(evaluator.BlinkDetectionEvaluator().evaluate(dataframe))
        f1_score = metrics.f1_score(y, predictions)
        avg_loss = np.mean(losses)

        print(metrics.classification_report(y, predictions))
        print(metrics.classification_report(y, moving_avg_pred))
        confussion_matrix = metrics.confusion_matrix(y, predictions).ravel()
        print(confussion_matrix)
        self.log("val_loss", avg_loss)
        self.log("val_f1", f1_score)

    def moving_average(self, values):
        #return np.append(np.average(sliding_window_view(values, window_shape = n), axis=1), np.zeros(n-1))
        return np.concatenate((np.zeros(1) ,np.average(sliding_window_view(values, window_shape = 3), axis=1), np.zeros(1)), axis=None)

    def moving_average_big(self, values):
        #return np.append(np.average(sliding_window_view(values, window_shape = n), axis=1), np.zeros(n-1))
        return np.concatenate((np.zeros(2) ,np.average(sliding_window_view(values, window_shape = 5), axis=1), np.zeros(2)), axis=None)



    def __fill_features_if_needed(self, features, targets):
        n = self.batch_size - features.size(dim=0)
        features = F.pad(features, (0, 0, 0, 0, 0, 0, 0, n), "constant", 0)
        targets = F.pad(targets, (0, n), "constant", 0)

        return features, targets

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5, verbose=True),
                "monitor": "val_f1"
            }
        }


train_dataset_dirs = [
    "/mnt/hdd/gcruz/eyesOriginalSize2/RN15Train",
    "/mnt/hdd/gcruz/eyesOriginalSize2/RN30Train",
    "/mnt/hdd/gcruz/eyesOriginalSize2/RN15Val",
    "/mnt/hdd/gcruz/eyesOriginalSize2/RN30Val"
]
# val_dataset_dirs = [
#      "/mnt/hdd/gcruz/eyesOriginalSize2/RN15Val",
#      "/mnt/hdd/gcruz/eyesOriginalSize2/RN30Val"
#  ]
val_dataset_dirs = [
    "/mnt/hdd/gcruz/eyesOriginalSize2/RN15Test",
    "/mnt/hdd/gcruz/eyesOriginalSize2/RN30Test"
]

callback = TuneReportCallback(
    {
        "loss": "val_loss",
        "f1": "val_f1"
    },
    on="validation_end")




def train_tune(config, epochs=10, gpus=[0]):
    image_size = config["image_size"]
    batch_size = config["batch_size"]

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
        #TransformerImgAugTransform(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    data = datamodule.VivitDataModule(train_dataset_dirs, val_dataset_dirs, train_transform, test_transform,
                                      batch_size)

    transformer = VivitTransformer(config, data.train_weights(), data)
    early_stop_callback = EarlyStopping(monitor="val_f1", mode="max", patience=15)
    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=gpus,
        #progress_bar_refresh_rate=0,
        num_sanity_val_steps=-1)
        #callbacks=[callback])
        #callbacks=[early_stop_callback])
    trainer.fit(transformer, datamodule=data)

def tune_vivit_asha(num_samples=10, epochs=10, gpus=[0]):
    config = {
        "image_size": tune.choice([64, 128]),
        "patch_size": tune.choice([8, 16]),
        "num_classes": tune.choice([2]),
        "num_frames": tune.choice([8, 16, 32, 64]),
        "dim": tune.choice([64, 128, 256, 512]),
        "batch_size": tune.choice([256, 512, 1024]),
        "depth": tune.randint(1, 4),
        "heads": tune.randint(2, 5),
        "in_channels": tune.choice([3]),
        "dim_head": tune.choice([32, 64, 128]),
        "dropout": tune.uniform(0, 0.2),
        #"emb_dropout": tune.uniform(0, 0.5),
        "dropout": tune.choice([0]),
        "emb_dropout": tune.choice([0]),
        "scale_dim": tune.choice([4]),
        "lr": tune.choice([0.0001, 0.00001, 0.000001]),
    }

    scheduler = ASHAScheduler(
        max_t=epochs,
        grace_period=3,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["image_size", "patch_size", "num_frames", "batch_size", "dim", "depth", "heads", "dim_head", "lr"],
        metric_columns=["loss", "f1", "training_iteration"])

    train_fn_with_parameters = tune.with_parameters(train_tune,
                                                    epochs=epochs,
                                                    gpus=gpus)

    analysis = tune.run(train_fn_with_parameters,
        metric="f1",
        mode="max",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        resources_per_trial={"cpu": 4, "gpu": 1},
        progress_reporter=reporter,
        name="tune_vivit_asha_2")

    print("Best hyperparameters found were: ", analysis.best_config)


# config_best = {
#     "image_size": 64,
#     "patch_size": 8,
#     "num_classes": 2,
#     "num_frames": 32,
#     "dim": 64,
#     "batch_size": 512,
#     "depth": 1,
#     "heads": 2,
#     "in_channels": 3,
#     "dim_head": 64,
#     "dropout": 0,
#     "emb_dropout": 0,
#     "scale_dim": 4,
#     "lr": 0.0001
# }

config_best_2 = {
    "image_size": 64,
    "patch_size": 16,
    #"image_size": 100,
    #"patch_size": 20,
    "num_classes": 2,
    "num_frames": 16,
    "dim": 256,
    "batch_size": 512,
    "depth": 1,
    "heads": 4,
    "in_channels": 3,
    "dim_head": 32,
    "dropout": 0,
    "emb_dropout": 0,
    "scale_dim": 4,
    "lr": 0.0001
}

#tune_vivit_asha(num_samples=30, epochs=15)
train_tune(config_best_2, epochs=300)

