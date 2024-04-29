import os
import optuna
from optuna.trial import TrialState
import torch.optim as optim
from tqdm import tqdm
from utils.tools import *
from utils.dataLoader import MyDataSet, dataset_collate
from torch.utils.data import DataLoader
from model.anchor_generate import generate_anchors
from model.anchor_match import multibox_target
from model.net import TinySSD
from model.loss import *
import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils

# ---------------------------------------------------------
# configuration information
# ---------------------------------------------------------
Dir_path = 'C:\\Users\\Marwan\\PycharmProjects\\TinySSD_Banana\\TinySSD_Banana'
voc_classes_path = os.path.join(Dir_path, 'model_data\\voc_classes.txt')
image_size_path = os.path.join(Dir_path, 'model_data\\image_size.txt')
train_file_path = '2077_train.txt'
val_file_path = '2077_val.txt'
anchor_sizes_path = os.path.join(Dir_path, 'model_data\\anchor_sizes.txt')
anchor_ratios_path = os.path.join(Dir_path, 'model_data\\anchor_ratios.txt')
iterations = 12000
batch_size = 128


def max_trial_callback(study, trial):
    n_complete = len([t for t in study.trials if
                      t.state == optuna.trial.TrialState.COMPLETE or t.state == optuna.trial.TrialState.RUNNING])
    if n_complete >= 100:
        study.stop()


def objective(trial):
    # ---------------------------------------------------------
    #                   Load training Data
    # ---------------------------------------------------------
    _, num_classes = get_classes(voc_classes_path)
    r = get_image_size(image_size_path)
    with open(train_file_path) as f:
        train_lines = f.readlines()
    train_dataset = MyDataSet(train_lines, r, mode='train')
    train_iter = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True,
                            drop_last=True,
                            collate_fn=dataset_collate)

    # ---------------------------------------------------------
    #                   Load validation Data
    # ---------------------------------------------------------
    with open(val_file_path) as f:
        val_lines = f.readlines()
    val_dataset = MyDataSet(val_lines, r, mode='validate')
    val_iter = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True,
                          drop_last=True,
                          collate_fn=dataset_collate)
    # --------------------------- ------------------------------
    #               Generate a prior anchor box
    # ---------------------------------------------------------
    sizes = get_anchor_info(anchor_sizes_path)
    ratios = get_anchor_info(anchor_ratios_path)
    if len(sizes) != len(ratios):
        ratios = [ratios[0]] * len(sizes)
    anchors_per_pixel = len(sizes[0]) + len(ratios[0]) - 1
    feature_map = [r // 8, r // 16, r // 32, r // 64, 1]
    anchors = generate_anchors(feature_map, sizes, ratios)  # (1600+400+100+25+1)*4 anchor boxes

    # ---------------------------------------------------------
    #                       Network Part
    # ---------------------------------------------------------
    param = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        "weight_decay": trial.suggest_float("weight_decay", 5e-5, 5e-1, log=True),
    }
    net = TinySSD(app=anchors_per_pixel, cn=num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    optimizer = getattr(optim, param["optimizer"])(net.parameters(), lr=param["learning_rate"], weight_decay=param["weight_decay"])
    scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # ---------------------------------------------------------
    #                       Start training
    # ---------------------------------------------------------
    num_epochs = 5  # (iterations // (len(train_dataset) // batch_size))
    anchors = anchors.to(device)
    training_loss = 0.0

    for epoch in range(num_epochs):
        net.train()
        for features, target in tqdm(train_iter):
            X, Y = features.to(device), target.to(device)  # (bs, 3, h, w) (bs, 100, 5)

            optimizer.zero_grad()
            # Predict the class and offset for each anchor box (multi-scale results are merged)
            cls_preds, bbox_preds = net(X)  # (bs, anchors, (1+c)) (bs, anchors*4)
            # Label the category and offset for each anchor box (bs, anchors*4) (bs, anchors*4) (bs, anchors)
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
            # Calculate loss function based on predicted and labeled values of class and offset
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            l.backward()
            optimizer.step()
            # training_metric.add(cls_eval(cls_preds, cls_labels), num_classes, bbox_eval(bbox_preds, bbox_labels, bbox_masks), num_classes)

        # validation of the model
        net.eval()
        validating_loss = 0.0
        with torch.no_grad():
            for features, target in tqdm(val_iter):
                X, Y = features.to(device), target.to(device)  # (bs, 3, h, w) (bs, 100, 5)

                # Predict the class and offset for each anchor box (multi-scale results are merged)
                cls_preds, bbox_preds = net(X)  # (bs, anchors, (1+c)) (bs, anchors*4)
                # Label the category and offset for each anchor box (bs, anchors*4) (bs, anchors*4) (bs, anchors)
                bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)

                # Calculate loss function based on predicted and labeled values of class and offset
                val_loss = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
                # val_loss = cls_eval(cls_preds, cls_labels) + bbox_eval(bbox_preds, bbox_labels, bbox_masks)
                validating_loss += val_loss.item()
        trial.report(validating_loss, epoch)

        scheduler_lr.step()
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss


if __name__ == '__main__':
    run = neptune.init(
        project="sexybroccoli/TinySSD",
        api_token="",
    )  # your credentials

    params = {"direction": "minimize", "n_trials": 20}
    run["parameters"] = params
    neptune_callback = optuna_utils.NeptuneCallback(run)

    study = optuna.create_study(study_name="TinySSD_HyperParameters", direction=params["direction"])
    study.optimize(lambda trial: objective(trial), n_trials=params["n_trials"], gc_after_trial=True, callbacks=[neptune_callback])

    run.stop()
    # study = optuna.create_study(study_name="TinySSD_HyperParameters", direction="minimize")
    # study.optimize(lambda trial: objective(trial), gc_after_trial=True, callbacks=[max_trial_callback])

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


