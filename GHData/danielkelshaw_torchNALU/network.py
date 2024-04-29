import os
import argparse
import pandas as pd

import torch
import torch.nn as nn

from models import MLP, NAC, NALU


arithmetic_functions = {
    'add': lambda x, y: x + y,
    'sub': lambda x, y: x - y,
    'mul': lambda x, y: x * y,
    'div': lambda x, y: x / y,
    'squared': lambda x: torch.pow(x, 2),
    'sqrt': lambda x: torch.sqrt(x)
}

models = {
    'None': None,
    'NAC': None,
    'NALU': None,
    'ReLU6': nn.ReLU6(),
    'Tanh': nn.Tanh(),
    'Sigmoid': nn.Sigmoid(),
    'Softsign': nn.Softsign(),
    'SELU': nn.SELU(),
    'ELU': nn.ELU(),
    'ReLU': nn.ReLU()
}


def generate_data(dim, fn, support):
    X = torch.FloatTensor(*dim).uniform_(*support)
    y = fn(*[X[:, i] for i in range(dim[1])]).unsqueeze(1)
    return X, y


def train(args, model, optimizer, criterion, data, target):

    for epoch in range(args.n_epochs):

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        m = torch.mean(torch.abs(target - output))

        loss.backward()
        optimizer.step()

        if epoch % args.log_interval == 0:
            print(f'Epoch {epoch:05}:\t'
                  f'Loss = {loss:.5f}\t'
                  f'MEA = {m:.5f}')


def test(model, data, target):

    with torch.no_grad():
        output = model(data)
        m = torch.mean(torch.abs(target - output))
        return m


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--n-layers', type=int, default=2, metavar='N',
                        help='number of layers (default: 2)')
    parser.add_argument('--hidden-dim', type=int, default=2, metavar='HD',
                        help='hidden dim (default: 2)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--n-epochs', type=int, default=10000, metavar='E',
                        help='number of training epochs (default: 1000)')
    parser.add_argument('--interp-support', type=list, default=[1, 100], metavar='S',
                        help='support for interpolation (default: [1, 100])')
    parser.add_argument('--extrap-support', type=list, default=[101, 200], metavar='S',
                        help='support for extrapolation (default: [101, 200])')
    parser.add_argument('--log-interval', type=int, default=500, metavar='LI',
                        help='train logging interval (default: 500)')
    parser.add_argument('--normalise', action='store_true', default=True,
                        help='normalise results (default: True)')

    args = parser.parse_args()

    # generate results directory
    save_dir = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    results = []
    for fn_type, fn in arithmetic_functions.items():

        if fn_type in ['squared', 'sqrt']:
            in_dim = 1
        else:
            in_dim = 2

        print(f'-> Testing function: {fn_type}')

        Xtrain, ytrain = generate_data(
            dim=(500, in_dim), fn=fn, support=args.interp_support
        )

        Xtest_interp, ytest_interp = generate_data(
            dim=(50, in_dim), fn=fn, support=args.interp_support
        )

        Xtest_extrap, ytest_extrap = generate_data(
            dim=(50, in_dim), fn=fn, support=args.extrap_support
        )

        print('-> Training random.')
        net = MLP(in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=1, n_layers=args.n_layers, act=None)

        random_mse_interp = torch.mean(
            torch.stack([test(net, Xtest_interp, ytest_interp)
                         for i in range(100)])
        ).item()

        random_mse_extrap = torch.mean(
            torch.stack([test(net, Xtest_extrap, ytest_extrap)
                         for i in range(100)])
        ).item()

        for name, model in models.items():

            if name == 'NAC':
                net = NAC(in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=1, n_layers=args.n_layers)
            elif name == 'NALU':
                net = NALU(in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=1, n_layers=args.n_layers)
            else:
                net = MLP(in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=1, n_layers=args.n_layers, act=model)

            print(f'-> Running: {name}')
            optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr)
            criterion = nn.MSELoss()
            train(args, net, optimizer, criterion, Xtrain, ytrain)

            interp_mse = test(net, Xtest_interp, ytest_interp).item()
            extrap_mse = test(net, Xtest_extrap, ytest_extrap).item()

            _tmp_interp = {
                'type': 'interp',
                'fn_type': fn_type,
                'activation': name,
                'mse': interp_mse,
                'random_mse': random_mse_interp
            }

            _tmp_extrap = {
                'type': 'extrap',
                'fn_type': fn_type,
                'activation': name,
                'mse': extrap_mse,
                'random_mse': random_mse_extrap
            }

            results.append(_tmp_interp)
            results.append(_tmp_extrap)

    # save results
    df_results = pd.DataFrame(results)

    df_results['normalised_mse'] = df_results.apply(
        lambda row: 100.0 * row['mse'] / row['random_mse'], axis=1
    )

    df_results.to_csv(os.path.join(save_dir, 'results.csv'))


if __name__ == '__main__':
    main()
