import argparse
import pandas as pd
from classifier import Transformer
import os
import datasets

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # ------------------------
    # TESTING ARGUMENTS
    # ------------------------
    parser.add_argument("--test_data", default=None, type=str, help="Path to the file containing the test data. Example: data/cola/test.tsv")
    parser.add_argument("--test_type", default="test", type=str, help="Type of the test.", choices=['test', 'validation'])
    parser.add_argument("--task_name", default="cola", type=str, help="GLUE task to be used.", choices=['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax'])
    parser.add_argument("--model", required=True, type=str, help="Path to the experiment model. Example: experiments/model")
    parser.add_argument("--output_file", default="result.tsv", required=False, type=str, help="Output file name. Example: submission/result.tsv")
    parser.add_argument("--label_form", default='names', type=str, help="Label form by class name or index.", choices=['index', 'names'])

    return parser.parse_args(args)

def load_model_from_experiment(experiment_folder: str):
    """ Function that loads the model from an experiment folder.
    :param experiment_folder: Path to the experiment folder.
    Return:
        - Pretrained model.
    """
    checkpoints = [
        file
        for file in os.listdir(experiment_folder)
        if file.endswith(".ckpt")
    ]
    checkpoint_path = experiment_folder + "/" + checkpoints[-1]

    model = Transformer.load_from_checkpoint(checkpoint_path)

    # Make sure model is in prediction mode
    model.eval()
    model.freeze()
    return model

def predict(row: dict, label_form: str, is_regression: bool):
    res = model.predict(row)

    if is_regression:
        pred = res['result']
    else:
        if label_form == 'names':
            pred = str(model.data.class_names[res['result']])
        else:
            pred = str(res['result'])
    return pred


if __name__ == "__main__":
    args = parse_args()

    print("Loading model...")
    model = load_model_from_experiment(args.model)


    print("Predict...")
    is_regression = args.task_name == "stsb"
    y_pred = []

    if args.test_data:
        if args.test_data.endswith('.csv'):
            df = pd.read_csv(args.test_data)
        elif args.test_data.endswith('.tsv'):
            df = pd.read_csv(args.test_data, sep='\t')
        elif args.test_data.endswith('.txt'):
            df = pd.read_fwf(args.test_data)

        for i, row in df.iterrows():
            y_pred.append(predict(row, args.label_form, is_regression))
        df['label'] = y_pred
    else:
        dataset = datasets.load_dataset('glue', args.task_name)
        df = pd.DataFrame()

        if args.test_type == "test":
            test_data = dataset['test']
        else:
            y_true = dataset['validation']['label']
            test_data = dataset['validation']

        df['index'] = [data['idx'] for data in test_data]
        for data in test_data:
            y_pred.append(predict(data, args.label_form, is_regression))

        if args.test_type == "validation":
            glue_metric = datasets.load_metric('glue', args.task_name)
            results = glue_metric.compute(predictions=y_pred, references=y_true)
            print(results)
            df['label'] = y_pred
            df=df[['index','label']]

    if args.output_file.endswith('.csv'):
        df.to_csv(args.output_file, index=False)
    elif args.output_file.endswith('.tsv'):
        df.to_csv(args.output_file, sep='\t', index=False)
