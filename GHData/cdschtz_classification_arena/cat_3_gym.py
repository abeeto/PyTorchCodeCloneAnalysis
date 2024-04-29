import pandas as pd
from simpletransformers.classification import ClassificationModel

from models.category_3 import ALL_CLASSIFICATION_MODELS
from utils import get_files_with_data


def get_dataset():
    all_files = get_files_with_data()
    data_frames = []
    for i, filepath in enumerate(all_files):
        tmp = pd.read_json(filepath, lines=True)
        data_frames.append(tmp)

    dataset = pd.concat(data_frames)  # has size of 265MB
    del data_frames

    # flatten meta column for sorting possibility after e.g. page views
    # -> label, title, text, id, input, touched, length, page views
    dataset = pd.concat([dataset.drop(['meta'], axis=1), dataset.meta.apply(pd.Series)], axis=1)

    # shuffle the dataset
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    dataset = dataset.drop(columns=['title', 'id', 'input', 'touched', 'length', 'pageviews'])
    dataset = dataset.reindex(columns=['text', 'label'])
    dataset = dataset.rename(columns={"label": "labels"})

    split = 331_028
    training_data = dataset.iloc[:split]
    test_data = dataset.iloc[split:]
    del dataset
    return training_data, test_data


def run(model_name=("distilbert", "distilbert-base-uncased")):
    # TODO: make directories in VM
    training_data, test_data = get_dataset()
    model = ClassificationModel(model_name[0], model_name[1])
    output_dir_train = "./saved_states/category3/" + model_name[0]
    output_dir_eval = "./results/category3/" + model_name[0]
    # create paths if they do not exist
    from pathlib import Path
    Path(output_dir_train).mkdir(parents=True, exist_ok=True)
    Path(output_dir_eval).mkdir(parents=True, exist_ok=True)
    model.train_model(training_data, args={"overwrite_output_dir": True}, output_dir=output_dir_train)
    result, model_outputs, wrong_predictions = model.eval_model(test_data, output_dir=output_dir_eval)


def run_all():
    for model_name in ALL_CLASSIFICATION_MODELS:
        run(model_name)


if __name__ == '__main__':
    run_all()
