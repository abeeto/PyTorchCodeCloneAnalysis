import yaml
from albumentations.core.serialization import from_dict

from collators import get_collator
from datasets import create_dataset, create_dataloader, create_sampler
from losses import create_loss
from lr_schedulers import create_lr_schedule
from lr_schedulers.compose import ComposeLrSchedulers
from metrics.metrics import create_metric, MetricsHandler
from models import create_model
from optimizers import create_optimizer


class TrainingManager:
    def __init__(self, path):
        self.path = path
        with open(path + "/" + "losses.yml") as file:
            self.losses = yaml.load(file.read(), Loader=yaml.FullLoader)
        with open(path + "/" + "model.yml") as file:
            self.models = yaml.load(file.read(), Loader=yaml.FullLoader)
        with open(path + "/" + "optimizer.yml") as file:
            self.optimizers = yaml.load(file.read(), Loader=yaml.FullLoader)
        with open(path + "/" + "params.yml") as file:
            self.params = yaml.load(file.read(), Loader=yaml.FullLoader)
        with open(path + "/" + "datasets.yml") as file:
            self.datasets = yaml.load(file.read(), Loader=yaml.FullLoader)
        try:
            with open(path + "/" + "trainer.yml") as file:
                self.trainer = yaml.load(file.read(), Loader=yaml.FullLoader)
        except FileNotFoundError:
            self.trainer = {}
            print(" no trainer params info")
        try:
            with open(path + "/" + "lr_scheduler.yml") as file:
                self.lr_schedulers = yaml.load(file.read(), Loader=yaml.FullLoader)
        except FileNotFoundError:
            print("no lr schedule info")
            self.lr_schedulers = None
        try:
            with open(path + "/" + "metrics.yml") as file:
                self.metrics = yaml.load(file.read(), Loader=yaml.FullLoader)
        except FileNotFoundError:
            print("no metrics info")
            self.metrics = None

    def loss(self):
        losses = {}
        loss_weights = {}
        for i, (loss, data) in enumerate(self.losses.items()):
            losses.update({loss: create_loss(data["func"], **data.get("kwargs", {}))})
            loss_weights.update({loss: data.get("weight", 1)})
        return losses, loss_weights

    def metric(self):
        if self.metrics:
            return MetricsHandler(
                {name: create_metric(name, **data.get("kwargs", {}) if data["kwargs"] else {}) for name, data in
                 self.metrics.items()})
        return MetricsHandler({})

    def model(self):
        if len(self.models) > 2:
            raise ValueError("more than one model in info")
        model_name, kwargs = next(iter(self.models.items()))
        return create_model(model_name, **kwargs)

    def teacher(self):
        if len(self.models) > 2:
            raise ValueError("more than one model in info")
        if len(self.models) < 2:
            return None
        model_name, kwargs = list(self.models.items())[1]
        return create_model(model_name, **kwargs)

    def lr_scheduler(self, optimizer):
        if self.lr_schedulers:
            if len(self.lr_schedulers) > 1:
                steps = []
                lrs = []
                schedulers = []
                for data in self.lr_schedulers:
                    if "start" not in data:
                        raise ValueError("lr schedule not starts")
                    elif "class" not in data:
                        raise ValueError("lr schedule class missing")
                    else:
                        steps.append(data["start"])
                        lrs.append(data.get("lr", []))
                        if not isinstance(lrs[-1], list):
                            lrs[-1] = [lrs[-1]]
                        if isinstance(optimizer.param_groups[0]["lr"], int):
                            lrs[-1] = [lrs[-1][p["lr"]] for p in optimizer.param_groups]
                        if lrs[-1] and 1 < len(lrs[-1]) != len(optimizer.param_groups):
                            raise ValueError("non consistent size of optimizer group params")
                        schedulers.append(create_lr_schedule(data["class"], optimizer, **data.get("kwargs", {})))
                return ComposeLrSchedulers(optimizer, schedulers, steps, lrs)
            else:
                scheduler_name, kwargs = next(iter(self.lr_schedulers.items()))
                return create_lr_schedule(scheduler_name, optimizer, **kwargs)
        return None

    def parameters(self, model):
        param = []
        if self.params:
            for name in self.params:
                part = False
                try:
                    a = model
                    for p in name.split("."):
                        if "[" in p:
                            n = p.split("[")
                            a = getattr(a, n[0])
                            a = a[int(n[1][:-1])]
                            part = True
                        else:
                            a = getattr(a, p)
                    if part:
                        param.append({"params": {key: val for key, val in a.named_parameters()}, "lr": 0.0})
                    else:
                        param.append({"params": dict(a.named_parameters()), "lr": 0.0})
                except AttributeError:
                    param.append({"params": getattr(model, name)(), "lr": 0.0})
            return param
        return model.named_parameters()

    def optimizer(self, parameters):
        if len(self.optimizers) > 1:
            raise ValueError("more than one optimizer in info")
        model_name, kwargs = next(iter(self.optimizers.items()))

        def get_decay(i):
            try:
                return kwargs["weight_decay"][i]
            except TypeError:
                return kwargs["weight_decay"]

        if "weight_decay" in kwargs:
            parameters = [{"params": param,
                           "weight_decay": get_decay(i)
                           if not name.endswith("bias") and param.requires_grad else 0.0,
                           "lr": i}
                          for i, params in enumerate(parameters) for name, param in params["params"]
                          ]
        else:
            parameters = [{"params": param,
                           "lr": i}
                          for i, params in enumerate(parameters) for name, param in params["params"]]
        return create_optimizer(model_name, parameters, **kwargs)

    def trainer_params(self):
        return self.trainer

    def dataloaders(self, **kwargs):
        dataloaders = {}
        for type, data in self.datasets.items():
            if "dataset" not in data or "dataloader" not in data:
                raise ValueError("dataset info missing for {%type}")
            dataset_args = {}
            if "transforms" in data:
                dataset_args.update({"transforms": from_dict(data["transform"])})
            else:
                dataset_args.update({"transforms": kwargs["transforms"][type] if "transforms" in kwargs and type in kwargs[
                    "transforms"] else None})
            if "folds" in kwargs and type in kwargs["folds"]:
                dataset_args.update({"fold": kwargs["folds"][type]})
            elif "folds" in data:
                dataset_args.update({"fold": data["folds"]})

            dataset = create_dataset(data["dataset"]["class"], **dataset_args,
                                     **data["dataset"].get("kwargs", {}))
            if "sampler" in data:
                sampler = create_sampler(data["sampler"]["class"], dataset, data["sampler"].get("kwargs", {}))
            else:
                sampler = None
            if "collate_fn" in data:
                collate = get_collator(data["collate_fn"])
            else:
                collate = None
            dataloader = create_dataloader(data["dataloader"]["class"], dataset, sampler, collate,
                                           **data["dataloader"].get("kwargs", {}))
            dataloaders.update({type: dataloader})
        return dataloaders
