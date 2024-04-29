from enum import Enum
import numpy as np
import pandas as pd

class Metric(Enum):
    ACCURACY = 'accuracy'
    SENSITIVITY = 'sensitivity'
    SPECITIFITY = 'specitifity'
    BALANCED_ACCURACY = 'balanced_accuracy'
    CM = 'print_cm'


class PerformanceEvaluation:
    def __init__(self, outputs, labels, num_labels):
        self.outputs = outputs
        self.labels = labels
        self.num_labels = num_labels

    def get_performance_metrics(self):
        return {
            Metric.ACCURACY: self.accuracy(),
            Metric.BALANCED_ACCURACY: self.balanced_accuracy(),
            Metric.SENSITIVITY: self.sensitivity(),
            Metric.SPECITIFITY: self.specitifity(),
            Metric.CM: self.print_cm()
        }

    # Define performance evaluation methods here
    def accuracy(self):
        predicted_labels = np.argmax(self.outputs, axis=1)
        return np.sum(predicted_labels == self.labels) / float(self.labels.size)

    def sensitivity(self):
        predicted_labels = np.argmax(self.outputs, axis=1)
        avg_sensitivity = 0.0
        for i in range(self.num_labels):
            predicted_labels_per_label = predicted_labels == i
            actual_labels_per_label = self.labels == i
            TP = np.sum(np.logical_and(predicted_labels_per_label == 1, actual_labels_per_label == 1))
            FN = np.sum(np.logical_and(predicted_labels_per_label == 0, actual_labels_per_label == 1))

            avg_sensitivity += 0.0 if (TP + FN) == 0 else TP / (TP + FN)

        return avg_sensitivity / self.num_labels


    def specitifity(self):
        predicted_labels = np.argmax(self.outputs, axis=1)
        avg_specitifity = 0.0
        for i in range(self.num_labels):
            predicted_labels_per_label = predicted_labels == i
            actual_labels_per_label = self.labels == i
            TN = np.sum(np.logical_and(predicted_labels_per_label == 0, actual_labels_per_label == 0))
            FP = np.sum(np.logical_and(predicted_labels_per_label == 1, actual_labels_per_label == 0))

            avg_specitifity += 0.0 if (TN + FP) == 0 else TN / (TN + FP)

        return avg_specitifity / self.num_labels

    def balanced_accuracy(self):
        predicted_labels = np.argmax(self.outputs, axis=1)
        avg_balanced_accuracy = 0.0
        for i in range(self.num_labels):
            predicted_labels_per_label = predicted_labels == i
            actual_labels_per_label = self.labels == i
            
            TP = np.sum(np.logical_and(predicted_labels_per_label == 1, actual_labels_per_label == 1))
            FN = np.sum(np.logical_and(predicted_labels_per_label == 0, actual_labels_per_label == 1))

            sensitivity = 0 if (TP + FN) == 0 else TP / (TP + FN)

            TN = np.sum(np.logical_and(predicted_labels_per_label == 0, actual_labels_per_label == 0))
            FP = np.sum(np.logical_and(predicted_labels_per_label == 1, actual_labels_per_label == 0))

            specitifity = 0 if (TN + FP) == 0 else TN / (TN + FP)

            avg_balanced_accuracy += (sensitivity + specitifity) / 2.0

        return avg_balanced_accuracy / self.num_labels

    def print_cm(self):
        predicted_labels = np.argmax(self.outputs, axis=1)
        df = pd.DataFrame(0, index=range(self.num_labels), columns=range(self.num_labels))
        for i in range(self.num_labels):
            for j in range(self.num_labels):
                predicted_labels_per_label = predicted_labels == i
                actual_labels_per_label = self.labels ==j
                df.iloc[i, j] = np.sum(np.logical_and(predicted_labels_per_label, actual_labels_per_label))

        return df