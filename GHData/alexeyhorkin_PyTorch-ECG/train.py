import torch
import argparse
import models
import test
import dataset as dts
import torch.nn as nn


def train(model, device, args, optimazer, criterior, train_dataset, test_dataset):
    for i in range(args.eph):
        optimazer.zero_grad()
        train_loss = 0
        for batch_ind, (input, target) in enumerate(train_dataset):
            input, target = input.to(device), target.to(device) 
            output = model(input)
            loss = criterior(output, target)
            loss.backward()
            optimazer.step()
            train_loss+=loss.item()
        print(f'Epoch {i+1}/{args.eph}')
        test.evaluate(model, test_dataset, device, criterior)
        print(f'Train loss is: {train_loss/len(train_dataset)}, \n')
        


def main(args, model):
    train_dataset, test_dataset = dts.get_dataset(args)
    optimazer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterior = nn.MultiLabelSoftMarginLoss()
    criterior = nn.BCELoss()
    input, target = next(iter(train_dataset))
    print(model(input.to(device)))
    train(model, device, args, optimazer, criterior, train_dataset, test_dataset)
    input, target = next(iter(train_dataset))
    print(model(input.to(device)))


if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=12, help='Size of batch during training')
    parser.add_argument('--num_workers', type=int, default=2, help='Count workers for data loading')
    parser.add_argument('--path_to_DataFile', type=str, default='ecg_data_200.json', help='Path to data file')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--eph', type=int, default=1, help='Count of epoches')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.CNN().to(device)
    main(args, model)

