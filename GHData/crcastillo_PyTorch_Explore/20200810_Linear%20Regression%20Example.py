"""
    PyTorch Explore - Simple Linear Regression Example

    Objective: Construct a Linear Regression within PyTorch framework and see if it can be stood up on GPU.

    Initial Build: 8/10/2020

    Notes:
    - I need to figure out how to normalize the input matrix otherwise the model is very sensitive to learning rate
        and can quickly result in missing the local minima
"""

# <editor-fold desc="Import relevant modules">

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold

from matplotlib import pyplot as plt
import plotly
from datetime import datetime
import pickle

import optuna
import multiprocessing

# </editor-fold>

# <editor-fold desc="Load dataset and split into Train/Test">

# Load regression dataset (boston) from sklearn
X, Y = load_boston(return_X_y = True)

# Split Train/Test
X_Train, X_Test, Y_Train, Y_Test = train_test_split(
    X
    , Y
    , test_size = 0.20
    , random_state = 123
)

# Set the device
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.get_device_name(Device)

# Set the number of processing cores and divide by 2
Cores = np.int(multiprocessing.cpu_count() / 2)

# </editor-fold>

# <editor-fold desc="Build LinearRegression Class and instantiate the model">

# Inherit the basic NeuralNetwork module
class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
    # Define the forward pass output
    def forward(self, X):
        out = self.linear(X)
        return out

# </editor-fold>

# <editor-fold desc="Define model parameters and train">

# Instantiate the model
LinearModel = LinearRegression(
    input_size = X_Train.shape[1]
    , output_size = 1
)

# If GPU available then instantiate model on GPU
LinearModel.to(Device)

# Define model parameters
epochs = 10000
learning_rate = 0.0000001
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(
    params = LinearModel.parameters()
    , lr = learning_rate
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer = optimizer
    , mode = 'min'
    , patience = 10
    , cooldown = 5
    , threshold = 1
)

# Store the initial start time
StartTime = datetime.now()

# Train the model
Train_MSE_Loss = [] # Instantiate MSE_Loss for model training
for _epoch in range(epochs):

    # Convert inputs and labels to Variable make sure to convert to float first
    _inputs = Variable(torch.from_numpy(X_Train).float().to(Device))
    _labels = Variable(torch.from_numpy(Y_Train).float().to(Device))

    # Store output from LinearModel as a function of inputs
    _outputs = LinearModel(_inputs)

    # Store the loss
    _loss = criterion(_outputs, _labels.unsqueeze_(1))
    Train_MSE_Loss += [_loss]

    # Clear gradient buffer from previous epochs, don't want to accumulate gradients
    optimizer.zero_grad()

    # Store gradient with respect to parameters
    _loss.backward()

    # Reduce learning rate if learning stagnates | TEST
    scheduler.step(_loss)

    # Update parameters
    optimizer.step()

    # Print it
    print('epoch {}, loss {}'.format(_epoch, _loss.item()))

# </editor-fold>

# <editor-fold desc="Test the model and plot predictions">

# Create predictions
with torch.no_grad():
    Pred_Y = LinearModel(Variable(torch.from_numpy(X_Test).float().to(Device))).cpu().data.numpy()

# Store Mean Squared Error
Test_MSE = np.square(np.subtract(Pred_Y.reshape(Pred_Y.shape[0], ), Y_Test)).mean()
print(Test_MSE)

# Display the runtime
print(datetime.now() - StartTime)

# Create Y_Test sorting index
Y_Test_idx = Y_Test.argsort()

# Plot performance figure
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(14, 10))
plt.plot(Y_Test[Y_Test_idx], 'go', label = 'Actual', alpha = 0.5)
plt.plot(Pred_Y[Y_Test_idx], '--', label = 'Predict', alpha = 0.5)
plt.legend(loc = 'best')
plt.show()

# </editor-fold>

# <editor-fold desc="Build LinearRegression Function and Train the model with Optuna">

# Split Train into X_Train_Split and X_Valid_Split
X_Train_Split, X_Valid_Split, Y_Train_Split, Y_Valid_Split = train_test_split(
    X_Train
    , Y_Train
    , test_size = 0.20
    , random_state = 123
)

# Create objective
def LinearRegression_Objective(trial):

    # Instantiate the model and send to GPU if available
    LinearModel = LinearRegression(
        input_size = X_Train_Split.shape[1]
        , output_size = 1
    ).to(Device)

    # Generate the optimizer search space
    optimizer_name = trial.suggest_categorical('optimizer', ['SGD', 'RMSprop', 'Adam'])
    # Generate the learning rate search space
    learning_rate = trial.suggest_loguniform('lr', 1e-9, 1e-3)
    # Generate the momentum search space
    if optimizer_name in ['SGD', 'RMSprop']:
        momentum = trial.suggest_uniform('momentum', 0.4, 0.99)

    # Define the optimizer differently for non-Adam vs Adam
    if optimizer_name in ['SGD', 'RMSprop']:
        optimizer = getattr(optim, optimizer_name)(
            params = LinearModel.parameters()
            , lr = learning_rate
            , momentum = momentum
        )
    else:
        optimizer = getattr(optim, optimizer_name)(
            params = LinearModel.parameters()
            , lr = learning_rate
        )

    # Instantiate MSELoss as the optimizer criterion
    criterion = torch.nn.MSELoss()

    # Train the model
    Train_MSE_Loss = []  # Instantiate MSE_Loss for model training
    for _epoch in range(epochs):

        # Convert inputs and labels to Variable make sure to convert to float first
        _inputs = Variable(torch.from_numpy(X_Train_Split).float().to(Device))
        _labels = Variable(torch.from_numpy(Y_Train_Split).float().to(Device))

        # Store output from LinearModel as a function of inputs
        _outputs = LinearModel(_inputs)

        # Store the loss
        _loss = criterion(_outputs, _labels.unsqueeze_(1))
        Train_MSE_Loss += [_loss]

        # Clear gradient buffer from previous epochs, don't want to accumulate gradients
        optimizer.zero_grad()

        # Store gradient with respect to parameters
        _loss.backward()

        # Update parameters
        optimizer.step()

        # Create predictions
        with torch.no_grad():
            Pred_Y = LinearModel(Variable(torch.from_numpy(X_Valid_Split).float().to(Device))).cpu().data.numpy()

        # Store Mean Squared Error
        Valid_MSE = np.square(np.subtract(Pred_Y.reshape(Pred_Y.shape[0], ), Y_Valid_Split)).mean()

        # Report progress
        trial.report(Valid_MSE, _epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Return the model performance metric
    return Valid_MSE


# Define model parameters
epochs = 10000

# Run the optimization
if __name__ == "__main__":
    # Instantiate the study
    LinearRegression_Study = optuna.create_study(
        direction = 'minimize'
        # , sampler = optuna.samplers.TPESampler
        , pruner = optuna.pruners.MedianPruner(
            n_startup_trials = 50
            , n_warmup_steps = 1000
            , interval_steps = 10
        )
    )
    # Start the optimization
    LinearRegression_Study.optimize(
        LinearRegression_Objective
        , n_trials = 200
        , n_jobs = Cores
    )

    # Store the pruned and complete trials
    pruned_trials = [t for t in LinearRegression_Study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in LinearRegression_Study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    # Print statistics
    print("Study statistics: ")
    print("  Number of finished trials: ", len(LinearRegression_Study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    # Store best_trial information and print it
    Best_Trial = LinearRegression_Study.best_trial
    print("Best trial:")
    print("  Value: ", Best_Trial.value)
    print("  Params: ")
    for key, value in Best_Trial.params.items():
        print("    {}: {}".format(key, value))

# Best Valid_MSE = 33.799386 - Second Run

# Pickle the LinearRegression_Study
pickle.dump(
    obj = LinearRegression_Study
    , file = open('Linear Regression Studies//20200820_LinearRegression_Study.sav', 'wb')
)

# Plot the optimization history
optuna.visualization.plot_optimization_history(LinearRegression_Study).show()
# Plot the parameter importance
optuna.visualization.plot_param_importances(LinearRegression_Study).show()
# Plot the parameter relationship
optuna.visualization.plot_contour(LinearRegression_Study).show()

# Save the trials_dataframe
LinearRegression_Study_df = LinearRegression_Study.trials_dataframe()

# Reconstruct optimization_history for optuna
# def OptimizationHistoryPlot(_study: optuna.study):
#
#     # Store layout
#     layout = plotly.graph_objs.Layout(
#         title = "Optimization History Plot"
#         , xaxis = {"title": "#Trials"}
#         , yaxis = {"title": "Objective Value"},
#     )
#
#     # Store trial
#     trials = [t for t in _study.trials if t.state == optuna.trial.TrialState.COMPLETE]
#
#     if len(trials) == 0:
#         optuna.logging.get_logger(__name__).warning("Study instance does not contain trials.")
#         return plotly.graph_objs.Figure(data = [], layout = layout)
#
#     best_values = [float("inf")] if _study.direction == optuna.study.StudyDirection.MINIMIZE else [-float("inf")]
#     comp = min if _study.direction == optuna.study.StudyDirection.MINIMIZE else max
#     for trial in trials:
#         trial_value = trial.value
#         assert trial_value is not None  # For mypy
#         best_values.append(comp(best_values[-1], trial_value))
#     best_values.pop(0)
#     traces = [
#         plotly.graph_objs.Scatter(
#             x = [t.number for t in trials],
#             y = [t.value for t in trials],
#             mode = "markers",
#             name = "Objective Value",
#         )
#         , plotly.graph_objs.Scatter(
#             x = [t.number for t in trials]
#             , y = best_values
#             , name = "Best Value"
#         ),
#     ]
#
#     # Store the final figure
#     figure = plotly.graph_objs.Figure(
#         data = traces
#         , layout = layout
#     )
#
#     # Return the final figure
#     return figure

# OptimizationHistoryPlot(LinearRegression_Study).show()


# </editor-fold>

# <editor-fold desc="Build LinearRegression Function w/ Cross Validation and Train the model with Optuna">

# Create LinearRegression with respect to multiple folds for cross-validation
def LinearRegression_CV_Objective(trial, Train_X_Fold, Train_Y_Fold, Valid_X_Fold, Valid_Y_Fold):

    # Instantiate the model and send to GPU if available
    LinearModel = LinearRegression(
        input_size=X_Train.shape[1]
        , output_size=1
    ).to(Device)

    # Generate the optimizer search space
    optimizer_name = trial.suggest_categorical('optimizer', ['SGD', 'RMSprop', 'Adam'])
    # Generate the learning rate search space
    learning_rate = trial.suggest_loguniform('lr', 1e-9, 1e-3)
    # Generate the momentum search space
    if optimizer_name in ['SGD', 'RMSprop']:
        momentum = trial.suggest_uniform('momentum', 0.4, 0.99)

    # Define the optimizer differently for non-Adam vs Adam
    if optimizer_name in ['SGD', 'RMSprop']:
        optimizer = getattr(optim, optimizer_name)(
            params = LinearModel.parameters()
            , lr = learning_rate
            , momentum = momentum
        )
    else:
        optimizer = getattr(optim, optimizer_name)(
            params = LinearModel.parameters()
            , lr = learning_rate
        )

    # Instantiate MSELoss as the optimizer criterion
    criterion = torch.nn.MSELoss()

    # Train the model
    Train_MSE_Loss = []  # Instantiate MSE_Loss for model training
    for _epoch in range(epochs):

        # Convert inputs and labels to Variable make sure to convert to float first
        _inputs = Variable(torch.from_numpy(Train_X_Fold).float().to(Device))
        _labels = Variable(torch.from_numpy(Train_Y_Fold).float().to(Device))

        # Store output from LinearModel as a function of inputs
        _outputs = LinearModel(_inputs)

        # Store the loss
        _loss = criterion(_outputs, _labels.unsqueeze_(1))
        Train_MSE_Loss += [_loss]

        # Clear gradient buffer from previous epochs, don't want to accumulate gradients
        optimizer.zero_grad()

        # Store gradient with respect to parameters
        _loss.backward()

        # Update parameters
        optimizer.step()

        # Create predictions
        with torch.no_grad():
            Pred_Y = LinearModel(Variable(torch.from_numpy(Valid_X_Fold).float().to(Device))).cpu().data.numpy()

        # Store Mean Squared Error
        Valid_MSE = np.square(np.subtract(Pred_Y.reshape(Pred_Y.shape[0], ), Valid_Y_Fold)).mean()

        # Report progress
        trial.report(Valid_MSE, _epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Return the model performance metric
    return Valid_MSE

# Create cross-validation objective
def CV_Objective(trial):

    # Create the Fold index
    Fold = KFold(
        n_splits = 5
        , shuffle = True
        , random_state = 10
    )

    # Instantiate list of Score
    Scores = []

    # Iterate for each fold
    for fold_idx, (train_idx, valid_idx) in enumerate(Fold.split(range(len(X_Train)))):

        # Create the various folds to feed into the model
        # Train_X_Fold = torch.utils.data.Subset(X_Train, train_idx)
        # Train_Y_Fold = torch.utils.data.Subset(Y_Train, train_idx)
        # Valid_X_Fold = torch.utils.data.Subset(X_Train, valid_idx)
        # Valid_Y_Fold = torch.utils.data.Subset(Y_Train, valid_idx)
        Train_X_Fold, Valid_X_Fold = X_Train[train_idx], X_Train[valid_idx]
        Train_Y_Fold, Valid_Y_Fold = Y_Train[train_idx], Y_Train[valid_idx]

        # Create model for each fold
        Valid_MSE = LinearRegression_CV_Objective(trial, Train_X_Fold, Train_Y_Fold, Valid_X_Fold, Valid_Y_Fold)

        # Print fold_idx and Valid_MSE message
        # print('CV_Fold = ' + str(fold_idx) + ' Valid_MSE = ' + str(Valid_MSE))

        # Append on the Valid_MSE from each run
        Scores.append(Valid_MSE)

    # Return back the average of the cross-validated score
    return np.mean(Scores)


# Define model parameters
epochs = 10000

# Run the optimization
if __name__ == "__main__":
    # Instantiate the study
    LinearRegression_CV_Study = optuna.create_study(
        direction = 'minimize'
        # , sampler = optuna.samplers.TPESampler
        , pruner = optuna.pruners.MedianPruner(
            n_startup_trials = 20
            , n_warmup_steps = 500
            , interval_steps = 10
        )
    )
    # Start the optimization
    LinearRegression_CV_Study.optimize(
        CV_Objective
        , n_trials = 200
        , n_jobs = Cores
    )

    # Store the pruned and complete trials
    pruned_trials = [t for t in LinearRegression_CV_Study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in LinearRegression_CV_Study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    # Print statistics
    print("Study statistics: ")
    print("  Number of finished trials: ", len(LinearRegression_CV_Study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    # Store best_trial information and print it
    Best_Trial = LinearRegression_CV_Study.best_trial
    print("Best trial:")
    print("  Value: ", Best_Trial.value)
    print("  Params: ")
    for key, value in Best_Trial.params.items():
        print("    {}: {}".format(key, value))

# Pickle the LinearRegression_CV_Study
pickle.dump(
    obj = LinearRegression_CV_Study
    , file = open('Linear Regression Studies//20200820_LinearRegression_CV_Study.sav', 'wb')
)

# </editor-fold>

# <editor-fold desc="Build LinearRegression_Test Function and test Validate vs CV-5 optimized models">

# Create objective
def LinearRegression_Test(Study: optuna.study.Study, X_Train, Y_Train, X_Test, Y_Test):

    # Instantiate the model and send to GPU if available
    LinearModel = LinearRegression(
        input_size = X_Train.shape[1]
        , output_size = 1
    ).to(Device)

    # Define the optimizer differently for non-Adam vs Adam
    if Study.best_params['optimizer'] in ['SGD', 'RMSprop']:
        optimizer = getattr(optim, Study.best_params['optimizer'])(
            params = LinearModel.parameters()
            , lr = Study.best_params['lr']
            , momentum = Study.best_params['momentum']
        )
    else:
        optimizer = getattr(optim, Study.best_params['optimizer'])(
            params = LinearModel.parameters()
            , lr = Study.best_params['lr']
        )

    # Instantiate MSELoss as the optimizer criterion
    criterion = torch.nn.MSELoss()

    # Train the model
    Train_MSE_Loss = []  # Instantiate MSE_Loss for model training
    for _epoch in range(epochs):

        # Convert inputs and labels to Variable make sure to convert to float first
        _inputs = Variable(torch.from_numpy(X_Train).float().to(Device))
        _labels = Variable(torch.from_numpy(Y_Train).float().to(Device))

        # Store output from LinearModel as a function of inputs
        _outputs = LinearModel(_inputs)

        # Store the loss
        _loss = criterion(_outputs, _labels.unsqueeze_(1))
        Train_MSE_Loss += [_loss]

        # Clear gradient buffer from previous epochs, don't want to accumulate gradients
        optimizer.zero_grad()

        # Store gradient with respect to parameters
        _loss.backward()

        # Update parameters
        optimizer.step()

    # Create predictions
    with torch.no_grad():
        Pred_Y = LinearModel(Variable(torch.from_numpy(X_Test).float().to(Device))).cpu().data.numpy()

    # Store Mean Squared Error
    Test_MSE = np.square(np.subtract(Pred_Y.reshape(Pred_Y.shape[0], ), Y_Test)).mean()

    # Return the model performance metric
    return Test_MSE

# Validate model Test_MSE = 28.195797
print(
    LinearRegression_Test(
        Study = LinearRegression_Study
        , X_Train = X_Train
        , Y_Train = Y_Train
        , X_Test = X_Test
        , Y_Test = Y_Test
    )
)

# CV-5 model Test_MSE = 28.191724
print(
    LinearRegression_Test(
        Study = LinearRegression_CV_Study
        , X_Train = X_Train
        , Y_Train = Y_Train
        , X_Test = X_Test
        , Y_Test = Y_Test
    )
)

# </editor-fold>