import numpy as np
import time
from collections import deque
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Step 1: Data Stream Generation
# This function generates a stream of data points. The data consists of a sine wave (seasonal pattern)
# combined with random noise to simulate real-world unpredictable data.
def generate_data_stream(length=1000, noise_factor=0.05, seasonal_period=100):
    """
    Simulates a real-time data stream.
    - length: Number of data points to generate.
    - noise_factor: Standard deviation of random noise added to the sine wave.
    - seasonal_period: Periodicity of the sine wave to simulate seasonality.
    """
    for i in range(length):
        # Generate a sine wave as the base value
        base_value = np.sin(2 * np.pi * (i % seasonal_period) / seasonal_period)
        # Add random Gaussian noise
        noise = np.random.normal(0, noise_factor)
        # Yield the data point (real-time streaming simulation)
        yield base_value + noise
        time.sleep(0.01)  # Simulate real-time delay between data points

# Step 2: Moving Average Calculation
# This function computes the moving average of the last 'window_size' data points.
def moving_average(window):
    """
    Calculates the moving average of the data in the window.
    - window: A list of the most recent data points.
    """
    return np.mean(window)  # Compute and return the average of the window

# Step 3: Anomaly Detection with Isolation Forest
# Isolation Forest is an unsupervised anomaly detection algorithm that isolates anomalies by
# constructing random decision trees. Points that are isolated early in the process are considered anomalies.
model = IsolationForest(contamination=0.05)  # 5% of data points are expected to be anomalies

def isolation_forest_anomaly_detection(window):
    """
    Detects anomalies using the Isolation Forest algorithm.
    Returns True if the most recent data point is an anomaly, False otherwise.
    """
    try:
        if len(window) == 0:
            raise ValueError("Window is empty, cannot perform anomaly detection.")

        # Reshape data to fit the Isolation Forest model
        data = np.array(window).reshape(-1, 1)
        # Fit the model on the sliding window of data
        model.fit(data)
        # Predict anomaly status (1: normal, -1: anomaly)
        predictions = model.predict(data)
        return predictions[-1] == -1  # Return True if the last point is an anomaly
    except Exception as e:
        print(f"Error in anomaly detection: {e}")
        return False  # Assume no anomaly if there's an error

# Step 4: Real-time Visualization with Anomalies
# This function monitors the data stream, applies moving average and anomaly detection, and visualizes
# the data points in real-time. Anomalies are highlighted on the plot with red dots.
def visualize_stream_with_anomalies(window_size=50):
    """
    Monitors the data stream in real-time, detects anomalies, and visualizes the stream with anomalies highlighted.
    - window_size: The size of the sliding window for moving average and anomaly detection.
    """
    # Initialize the data stream
    stream = generate_data_stream()
    data_window = deque(maxlen=window_size)  # Store the last 'window_size' data points
    all_data = []  # Store all data points for visualization
    anomalies = []  # Store the indices of detected anomalies for visualization

    plt.ion()  # Enable interactive mode for real-time plotting
    fig, ax = plt.subplots(figsize=(10, 6))  # Set up the figure for plotting

    # Iterate through the data stream
    for i, data in enumerate(stream):
        data_window.append(data)  # Add new data point to the window
        all_data.append(data)  # Store all data points

        # When the window is full, start processing
        if len(data_window) == window_size:
            avg = moving_average(data_window)  # Calculate the moving average

            # Detect if the most recent point is an anomaly
            if isolation_forest_anomaly_detection(data_window):
                anomalies.append(i)  # Record the index of the anomaly
                print(f"Anomaly detected at index {i}: {data:.4f}, Moving Average: {avg:.4f}")

            # Update the plot
            ax.clear()  # Clear the previous plot
            ax.plot(all_data, label="Data Stream")  # Plot the data stream

            # Highlight anomalies with red dots
            ax.scatter(anomalies, [all_data[j] for j in anomalies], color='red', label="Anomalies")

            # Add plot details
            ax.set_title("Data Stream with Anomalies")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend()

            # Pause to refresh the plot in real-time
            plt.pause(0.01)

        # Stop after plotting 500 data points for the sake of visualization
        if i > 500:
            break

    plt.ioff()  # Disable interactive mode
    plt.show()  # Display the final plot

# Step 5: Run the real-time visualization and anomaly detection
if __name__ == "__main__":
    visualize_stream_with_anomalies()
