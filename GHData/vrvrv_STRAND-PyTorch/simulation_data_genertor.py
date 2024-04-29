import os
import json
import h5py
import pickle
import argparse
from tqdm import tqdm
from src.generator import SimulationGenerator


def gen_single_simulation_data(rank, n, m, nbinom, disp_param):
    g = SimulationGenerator(
        rank=rank,
        dim=10,
        context_dims=[3, 3, 16, 4, 2],
        mutation=96,
        total_sample=n,
        random_theta_generation=True,
        nbinom=nbinom,
        disp_param=disp_param
    )

    return g.sample(m=m)


def generate_data(args):
    # n_range = [10, 100, 500, 1000, 2000, 3000]
    m_range = [10, 100, 1000, 10000]
    # r_range = [5]
    n_range = [10, 100, 1000, 3000]
    # m_range = [50]

    SAVE_DIR = f"data/simulation_{args.id}"

    if os.path.isdir(SAVE_DIR):
        raise NameError(f"Already exists the directory, {SAVE_DIR}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    with tqdm(total=len(n_range) * len(m_range),
              desc=f'Simulation data generation id: {args.id}') as pbar:
        for row, n in enumerate(n_range):
            g = SimulationGenerator(
                rank=args.rank,
                dim=10,
                context_dims=[3, 3, 16, 4, 2],
                mutation=96,
                total_sample=n,
                random_theta_generation=True,
                nbinom=args.nbinom,
                disp_param=args.disp_param
            )
            for col, m in enumerate(m_range):
                data, param = g.sample(m=m)

                with h5py.File(os.path.join(SAVE_DIR, f'rank_{args.rank}_m_{m}_n_{n}.hdf5'), 'w') as f:
                    f['count_tensor'] = data['count_tensor']
                    f['feature'] = data['feature']

                with open(os.path.join(SAVE_DIR, f'rank_{args.rank}_m_{m}_n_{n}_param.pkl'), 'wb') as f:
                    pickle.dump(param, f)

                pbar.update(1)

    meta = {
        'rank': args.rank,
        'n_range': n_range,
        'm_range': m_range,
        'distribution': {
            'family': 'nbinom' if args.nbinom else 'poisson',
            'disp_param': args.disp_param if args.nbinom else None
        }
    }

    with open(os.path.join(SAVE_DIR, 'meta.json'), "w") as f:
        json.dump(meta, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data prepare')
    parser.add_argument('--id', type=int, default=1)
    parser.add_argument('--rank', type=int, default=5)
    parser.add_argument('--nbinom', action='store_true')
    parser.add_argument('--disp_param', type=float, default=50)
    args = parser.parse_args()

    generate_data(args)
