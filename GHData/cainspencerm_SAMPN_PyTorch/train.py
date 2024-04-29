import models, schedule, utils, dataset as ds  # local imports

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.utils import tensorboard as tb
import argparse


def main():
    parser = argparse.ArgumentParser(description='Train a model on a dataset.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size.')
    parser.add_argument('--warmup-epochs', type=float, default=2, help='Number of epochs to warmup.')
    parser.add_argument('--dataset', type=str, default='lipophilicity', help='Dataset type (lipophilicity or solubility).')
    parser.add_argument('--model', type=str, default='QSAR', help='Model type (QSAR or QSARPlus).')
    args = parser.parse_args()

    # Initialize the metrics.
    criterion = nn.MSELoss()
    metric_funcs = list(utils.get_metrics(args.model).values())

    device = torch.device('cpu')

    # Create the model.
    if args.model == 'QSAR':
        model = models.QSAR()
        utils.initialize_weights(model)
    elif args.model == 'QSARPlus':
        model = models.QSARPlus()
        args.batch_size = 1  # QSARPlus requires batch size of 1.
    else:
        raise ValueError('Invalid model type: {}'.format(args.model))

    model.to(device)

    # Prepare dataset and dataloaders.
    if args.dataset == 'lipophilicity':
        data_path = './data/LogP_moleculenet.csv'
    elif args.dataset == 'solubility':
        data_path = './data/water_solubilityOCD.csv'
    else:
        raise ValueError('Invalid dataset type: {}'.format(args.dataset))

    train_set = ds.DGLDataset(data_path, split='train')
    val_set = ds.DGLDataset(data_path, split='valid')
    test_set = ds.DGLDataset(data_path, split='test')

    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, num_workers=4)

    # Create the optimizer and scheduler.
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.)
    if args.model == 'QSAR':
        init_lr, max_lr, final_lr = 1e-4, 1e-3, 1e-4
        scheduler = schedule.NoamLR(
            optimizer=optimizer,
            warmup_epochs=[args.warmup_epochs],
            total_epochs=[args.epochs],
            steps_per_epoch=len(train_loader),
            init_lr=[init_lr],
            max_lr=[max_lr],
            final_lr=[final_lr]
        )

    # Create the tensorboard writer.
    writer = tb.SummaryWriter('./runs/{}_{}'.format(args.dataset, args.model))

    # Begin training.
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):

        # Train the model.
        model.train()
        train_loss = 0.
        for smiles, labels in train_loader:
            labels = labels.to(device)

            # Zero the gradients.
            optimizer.zero_grad()

            # Forward pass.
            preds = model(smiles, device=device)
            loss = criterion(preds, labels)

            # Backward pass.
            loss.backward()
            optimizer.step()
            if args.model == 'QSAR':
                scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Validate the model.
        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for smiles, labels in val_loader:
                labels = labels.to(device)
                
                preds = model(smiles, device=device)

                val_loss += criterion(preds, labels).item()

        val_loss /= len(val_loader)
        writer.add_scalar('Loss/val', val_loss, epoch)

        print(f'Epoch {epoch} - Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}')

        # Save the model if the validation score is the best.
        if val_loss < best_loss:
            best_loss = val_loss

            torch.save(model.state_dict(), f'./checkpoint/{args.dataset}/{args.model}_state_dict.pt')

        writer.flush()

    writer.close()

    # Test the best model.
    model = model.to(device)
    model.load_state_dict(torch.load(f'./checkpoint/{args.dataset}/{args.model}_state_dict.pt'))

    model.eval()
    scores = np.zeros(len(metric_funcs))
    with torch.no_grad():
        for smiles, labels in test_loader:
            labels = labels.to(device)

            preds = model(smiles, device=device)

            # Calculate metrics.
            for idx, metric_func in enumerate(metric_funcs):
                scores[idx] += metric_func(preds, labels)

    scores /= len(test_loader)
    for score, metrics_name in zip(scores, utils.get_metrics(args.model).keys()):
        print(f'Model test {metrics_name} = {score:.6f}')


if __name__ == '__main__':
    main()
