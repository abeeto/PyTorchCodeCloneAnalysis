"""filename: prune.py"""

import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection

import optuna


def objective(trial):
    iris = sklearn.datasets.load_iris()
    classes = list(set(iris.target))
    size = trial.suggest_float('size', 0.01, 0.9)
    train_x, valid_x, train_y, valid_y = \
        sklearn.model_selection.train_test_split(iris.data, iris.target, test_size=size, random_state=0)

    alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-1)
    clf = sklearn.linear_model.SGDClassifier(alpha=alpha)

    for step in range(100):
        clf.partial_fit(train_x, train_y, classes=classes)

        # Report intermediate objective value.
        intermediate_value = 1.0 - clf.score(valid_x, valid_y)
        trial.report(intermediate_value, step)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

    return 1.0 - clf.score(valid_x, valid_y)


# Set up the median stopping rule as the pruning condition.
sampler = optuna.samplers.CmaEsSampler()
study = optuna.create_study(pruner=optuna.pruners.MedianPruner(), sampler=sampler)
study.optimize(objective, n_trials=100)