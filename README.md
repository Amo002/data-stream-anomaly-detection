# Data Stream Anomaly Detection

## Project Overview
This project implements real-time anomaly detection in a continuous data stream using Python. The data stream consists of a simulated sine wave with added random noise to mimic real-world behavior. We apply a moving average for trend analysis and the **Isolation Forest** algorithm for anomaly detection. The project also includes real-time visualization of the data stream and flagged anomalies.

## Key Features
- **Real-time Data Stream**: Simulates a continuous data stream with regular patterns and random noise.
- **Moving Average Calculation**: Smooths the data using a sliding window to detect trends.
- **Isolation Forest Anomaly Detection**: Identifies unusual data points (anomalies) based on the Isolation Forest machine learning algorithm.
- **Real-time Visualization**: Displays the data stream in real-time with anomalies highlighted on a dynamic plot.

## How It Works
1. **Data Stream Simulation**: The data stream is generated using a sine wave with added Gaussian noise to simulate real-world systems with periodic behavior and randomness.
2. **Moving Average**: The moving average smooths out fluctuations in the data, providing a clearer view of the underlying trend.
3. **Isolation Forest Anomaly Detection**: The Isolation Forest algorithm is applied to a sliding window of recent data points to flag anomalies in the data stream.
4. **Visualization**: The data is plotted in real-time, with detected anomalies highlighted using red markers for quick identification.

## Algorithm Selection
We chose **Isolation Forest** for anomaly detection due to its efficiency in detecting outliers in high-dimensional data and its ability to handle concept drift in streaming data. The moving average is used for trend smoothing and allows for a more accurate detection of deviations.

### Why Isolation Forest?
- **Unsupervised**: No need for labeled data; works well with new, unseen data in real time.
- **Efficiency**: Isolation Forest is fast and lightweight, making it ideal for streaming data applications.
- **Concept Drift**: By analyzing data within a sliding window, Isolation Forest adapts to changing trends in the data over time.

## Requirements
The project requires the following Python libraries:
- `matplotlib`
- `numpy`
- `scikit-learn`

To install the dependencies, run:

```bash
pip install -r requirements.txt
