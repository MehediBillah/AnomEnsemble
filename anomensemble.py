import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


class AnomEnsemble:
    """
    An ensemble anomaly detection system which uses multiple models
    to identify outliers in a dataset. The ensemble uses majority voting
    among different models to predict anomalies.
    """
    def __init__(self, models_params):
        self.models_params = models_params
        self.models = []
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def split_sequence(self, sequence, window_size):
        # Splits the sequence into sub-sequences of the given window size.
        return [sequence[i:(i + window_size)] for i in range(len(sequence) - window_size + 1)]

    def extract_features(self, data, window_size):
        # Extracts statistical features from the data within a sliding window.
        temp_data = self.split_sequence(data, window_size)
        features = []
        for func in [np.mean, np.median, np.std, np.max, np.min, np.var]:
            features.append([func(window) for window in temp_data])
        return np.column_stack(features)

    def fit(self, data, window_size):
        # Fits the ensemble models to the training data.
        features = self.extract_features(data, window_size)
        scaled_features = self.scaler.fit_transform(features)

        for name, params in self.models_params.items():
            model = params['model'](**params['params']).fit(scaled_features)
            self.models.append((name, model))
        return self

    def predict(self, data, window_size):
        # Predicts the anomalies on the given data.
        features = self.extract_features(data, window_size)
        scaled_features = self.scaler.transform(features)
        predictions = np.array([model.predict(scaled_features) for _, model in self.models])
        return np.apply_along_axis(lambda x: np.bincount(x + 1, minlength=2).argmax() - 1, axis=0, arr=predictions)

    def plot_results(self, data, window_size, anomalies_indices):
        # Plots the data and the identified anomalies.
        plt.figure(figsize=(15, 5))
        plt.plot(data, label='Data')
        plt.scatter(anomalies_indices, data[anomalies_indices], color='r', label='Anomaly', zorder=3)
        plt.title('Anomaly Detection with AnomEnsemble')
        plt.legend()
        plt.show()


# Usage
if __name__ == "__main__":
    window_size = 5  # Window size for feature extraction
    num_samples = 1000  # Number of data samples

    # Define the models and their parameters to include in the ensemble
    models_params = {
        'SVM': {'model': OneClassSVM, 'params': {'nu': 0.05}},
        'LOF': {'model': LocalOutlierFactor, 'params': {'contamination': 0.05, 'novelty': True}},
        'IForest': {'model': IsolationForest, 'params': {'contamination': 0.05}}
    }

    # Create an instance of AnomEnsemble
    anomaly_detector = AnomEnsemble(models_params)

    # Assume `data` is the input data array
    data = np.random.random(num_samples)

    # Fit the anomaly detector to the data
    anomaly_detector.fit(data, window_size)

    # Predict anomalies on the data
    anomalies = anomaly_detector.predict(data, window_size)

    # Identify the indices of the anomalies
    anomalies_indices = np.where(anomalies == -1)[0] + window_size - 1

    # Visualize the anomalies on the plot
    anomaly_detector.plot_results(data, window_size, anomalies_indices)
