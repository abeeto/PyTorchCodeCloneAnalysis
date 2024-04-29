import argparse
import os
import pickle
from ast import literal_eval

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description="NGD Benchmarking Result Processor")
parser.add_argument(
    "--RESULT_SAVE_FOLDER", "-r", type=str, default="/gpfs/fs0/data/DeepLearning/sabedin/Data/ngd-benchmark/results",
    required=False, help="folder to save results"
)
parser.add_argument("--RUN_IDS", "-ids", type=dict,
                    default={'gpu_gpfs': '2021_09_08-10_46_33', 'gpu_sdc': '2021_09_08-10_51_45'}, required=False,
                    help="Dict")
args = parser.parse_args()


def get_plot_df(name, run_id):
    # Read all the files for the run ID
    with open(os.path.join(args.RESULT_SAVE_FOLDER, run_id + '_config.pkl'), 'rb') as f:
        config = pickle.load(f)
    # print(json.dumps(config, sort_keys=True, indent=4))
    train_result_df = pd.read_csv(os.path.join(args.RESULT_SAVE_FOLDER, run_id + '_training.csv'),
                                  converters={'durations': literal_eval})
    inference_result_df = pd.read_csv(os.path.join(args.RESULT_SAVE_FOLDER, run_id + '_inference.csv'),
                                      converters={'durations': literal_eval})

    # Average duration
    train_result_df[name + '_average_duration'] = train_result_df['durations'].apply(lambda x: sum(x) / len(x))
    inference_result_df[name + '_average_duration'] = inference_result_df['durations'].apply(lambda x: sum(x) / len(x))

    train_total_df = train_result_df[['model_name', 'total_duration']]
    train_total_df = train_total_df.rename(columns={'total_duration': name + '_total_duration'})
    inference_total_df = inference_result_df[['model_name', 'total_duration']]
    inference_total_df = inference_total_df.rename(columns={'total_duration': name + '_total_duration'})

    train_result_df = train_result_df[['model_name', name + '_average_duration']]
    inference_result_df = inference_result_df[['model_name', name + '_average_duration']]

    return train_result_df, inference_result_df, train_total_df, inference_total_df


if __name__ == "__main__":

    plot_df = pd.DataFrame(columns=['model_name'])
    # Loop thru to collect all the average times
    for key, value in args.RUN_IDS.items():
        train_result_df, inference_result_df, train_total_df, inference_total_df = get_plot_df(key, value)

        plot_df = plot_df.merge(inference_total_df, on='model_name', how='outer')

    # Plot
    y_axis_col_names = plot_df.columns.values.tolist()
    print(y_axis_col_names)
    plot_df.plot(x="model_name", y=y_axis_col_names[1:], kind="bar")
    plt.xlabel("Model Name")
    plt.ylabel("Time(ms)")
    plt.show()
