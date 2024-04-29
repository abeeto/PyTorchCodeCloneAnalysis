import os
from tqdm import tqdm
from utils.tools import *
from utils.dataLoader import MyDataSet, dataset_collate
from torch.utils.data import DataLoader
from model.anchor_generate import generate_anchors
from model.anchor_match import multibox_target
from model.net import TinySSD
from model.loss import *

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
batch_size = 64


def train():
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
    net = TinySSD(app=anchors_per_pixel, cn=num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # noinspection PyBroadException
    try:
        net.load_state_dict(torch.load(os.path.join(Dir_path, 'model_data\\result.pt')))
        print("Fine-Tuning...")
    except:
        print("Training from scratch...")

    trainer = torch.optim.Adam(net.parameters(), lr=0.0037, weight_decay=0.0011)
    # trainer = torch.optim.Adam(net.parameters(), lr=0.03, weight_decay=5e-5)

    # scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(trainer, 100)
    scheduler_lr = torch.optim.lr_scheduler.StepLR(trainer, step_size=1, gamma=0.1)
    # ---------------------------------------------------------
    #                       Start training
    # ---------------------------------------------------------
    num_epochs, timer = (iterations // (len(train_dataset) // batch_size)), Timer()
    print(f' epochs: {num_epochs}')
    timer.start()
    # animator = Animator(xlabel='Epoch', xlim=[1, num_epochs], legend=['Class Error', 'Bbox MeanAvgError'])
    animator = Animator(xlabel='Epoch', xlim=[1, num_epochs], legend=['t-loss', 'v-loss'])
    net = net.to(device)
    anchors = anchors.to(device)
    # training_cls_loss, training_bbox_loss = None, None
    # validating_cls_loss, validating_bbox_loss = None, None
    for epoch in range(num_epochs):
        print(f' learning rate: {scheduler_lr.get_last_lr()}')
        # training_metric = Accumulator(4)
        # validating_metric = Accumulator(4)
        net.train()
        # training_loss = 0.0
        for features, target in tqdm(train_iter):
            trainer.zero_grad()
            X, Y = features.to(device), target.to(device)  # (bs, 3, h, w) (bs, 100, 5)

            # Predict the class and offset for each anchor box (multi-scale results are merged)
            cls_preds, bbox_preds = net(X)  # (bs, anchors, (1+c)) (bs, anchors*4)
            # Label the category and offset for each anchor box (bs, anchors*4) (bs, anchors*4) (bs, anchors)
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)

            # Calculate loss function based on predicted and labeled values of class and offset
            train_loss = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            train_loss.backward()
            trainer.step()
            # training_loss += train_loss.item()
            # training_metric.add(cls_eval(cls_preds, cls_labels), 1, bbox_eval(bbox_preds, bbox_labels, bbox_masks), 1)

        net.eval()
        validating_loss = 0.0
        for features, target in tqdm(val_iter):
            X, Y = features.to(device), target.to(device)  # (bs, 3, h, w) (bs, 100, 5)
            with torch.no_grad():
                # Predict the class and offset for each anchor box (multi-scale results are merged)
                cls_preds, bbox_preds = net(X)  # (bs, anchors, (1+c)) (bs, anchors*4)
                # Label the category and offset for each anchor box (bs, anchors*4) (bs, anchors*4) (bs, anchors)
                bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)

                # Calculate loss function based on predicted and labeled values of class and offset
                # val_loss = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
                cls_loss = cls_eval(cls_preds, cls_labels)
                bbox_loss = bbox_eval(bbox_preds, bbox_labels, bbox_masks)
                val_loss = cls_loss + bbox_loss
                validating_loss += val_loss.item()

        # learning rate decay
        scheduler_lr.step()

        # reserved for display
        # training_cls_loss, training_bbox_loss = training_metric[0] / training_metric[1], training_metric[2] / training_metric[3]
        # validating_cls_loss, validating_bbox_loss = validating_metric[0] / validating_metric[1], validating_metric[2] / validating_metric[3]

        animator.add(epoch + 1, validating_loss)
        print(f'epoch {epoch + 1}/{num_epochs}: ', ' v-loss', validating_loss)

        # animator.add(epoch + 1, (training_cls_loss, training_bbox_loss))
        # print(f'epoch {epoch + 1}/{num_epochs}: ', 't-cls-loss: ', training_cls_loss, ' t-box-loss', training_bbox_loss)

        # Save the trained model for each epoch
        torch.save(net.state_dict(), f'model_data/result_{epoch + 1}.pt')

    print(f'validation loss {validating_loss:.2e}')
    # print(f'class loss {validating_cls_loss:.2e}, bbox loss {validating_bbox_loss:.2e}')
    print(f'total time: {timer.stop():.1f}s', f' on {str(device)}')


if __name__ == '__main__':
    train()
