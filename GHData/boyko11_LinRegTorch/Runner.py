from service.data_service import DataService
from service.plot_service import PlotService
from linear_regression_learner import LinearRegressionLearner
import numpy as np
import torch


class Runner:

    def __init__(self):
        num_features = 3
        self.linear_regression_learner = LinearRegressionLearner(theta_size=num_features, learning_rate=0.000023)

    def run(self):

        data = DataService.load_csv("data/blood_pressure_cengage.csv")
        feature_data = data[:, :-1]
        labels = data[:, -1]

        #add one for the bias
        feature_data = np.insert(feature_data, 0, 1, axis=1)
        features_tensor = torch.tensor(feature_data.astype(np.float32))
        labels_tensor = torch.tensor(labels.astype(np.float32))

        train_epochs = 500
        self.linear_regression_learner.train(features_tensor, labels_tensor, epochs=train_epochs)

        trained_predictions = self.linear_regression_learner.predict(features_tensor)

        print("|Age|Weight|Actual Blood Pressure|Predicted Blood Pressure|Loss|")
        print("|:-------:|:---:|:--------------------:|:---------------------:|")

        train_records = data[:, :-1]
        predicted_labels = []
        for record_index, prediction in enumerate(trained_predictions):
            age = train_records[record_index, 0]
            weight = train_records[record_index, 1]
            label = labels[record_index]
            predicted_label = np.rint(prediction.data.numpy()[0])
            predicted_labels.append(predicted_label)
            loss = np.abs(label - predicted_label)

            print("|{0}|{1}|{2}|{3}|{4}|".format(age, weight, label, predicted_label, loss))

        PlotService.plot_line(
            x=np.arange(train_epochs),
            y=self.linear_regression_learner.loss_history,
            x_label="Torch Epochs",
            y_label="Loss",
            title="Absolute Loss per Epoch")

        PlotService.plot3d_scatter_compare(train_records, labels, trained_predictions.data.numpy(),
                                           labels=['Age', 'Weight', 'BP'],
                                           title="Actual vs Projected")


if __name__ == "__main__":

    Runner().run()
