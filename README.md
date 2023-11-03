# AnomEnsemble for Injection Molding Defect Detection

AnomEnsemble is a sophisticated anomaly detection system designed to identify real-time defects in injection molding machines. The system leverages an ensemble approach, utilizing OneClassSVM, LocalOutlierFactor, and IsolationForest models to ensure high accuracy and reliability in detecting anomalies that could indicate potential defects.

## Description

Injection molding is a critical process in manufacturing, and the timely detection of defects can save significant costs and prevent the production of substandard products. AnomEnsemble detects deviations in operational data that may signal the onset of machine defects, allowing for prompt intervention. This repository contains a simplified, dummy version of the original AnomEnsemble system, aimed at demonstrating the underlying principles and functionality.

## Features

- Real-time anomaly detection tailored for injection molding machines.
- Ensemble method combining multiple machine learning models for robust predictions.
- Customizable feature extraction to suit different machine behaviors and data types.
- Visualization capabilities for quick identification of potential issues.
- Adaptable to different machine learning scenarios beyond injection molding.

## Installation

To get started with AnomEnsemble, clone the repository and install the necessary dependencies:

```sh
git clone https://github.com/MehediBillah/AnomEnsemble.git
cd AnomEnsemble
pip install -r requirements.txt
```

## Usage

Use AnomEnsemble by integrating it with your data acquisition systems to analyze real-time data from your injection molding machines. Set up is straightforward:

```python
from anomensemble import AnomEnsemble

# Initialize the anomaly detection system
detector = AnomEnsemble()

# Real-time data monitoring and fitting
detector.fit(real_time_data_stream, window_size=5)

# Continuously predict and monitor for anomalies
while True:
    current_data = get_current_data()  # Implement this function based on your data source
    anomaly_status = detector.predict(current_data, window_size=5)
    if anomaly_status == -1:
        alert_system()  # Implement an alert system

# Optionally, visualize the detection results
detector.plot_results(data_stream, anomalies_indices)
```

## Contributing

We welcome contributions that can improve the accuracy, efficiency, and usability of AnomEnsemble. Please follow the standard contribution guidelines as outlined in the repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- Project Maintainer: Muhammad Mehedi Billah
- Email: contact@billah.tech
- Project Repository: [https://github.com/MehediBillah/AnomEnsemble](https://github.com/MehediBillah/AnomEnsemble)

Feel free to reach out if you have any questions, or would like to contribute to the project.
```
