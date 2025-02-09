"""
This application:
  1. Loads and preprocesses the Kaggle Credit Card Fraud Detection dataset.
  2. Trains a deep autoencoder using only normal transactions (Class == 0).
  3. Computes a reconstruction error threshold from training data.
  4. Simulates streaming by evaluating a (mixed) sample of transactions in real time.
  5. Uses a simple rule engine: if reconstruction error > threshold, flag as anomaly.
  6. Displays results in a Tkinter GUI.
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense

# --- Block 1: File Paths and Setup ---
DATA_PATH = os.path.join("data", "creditcard.csv")
MODEL_PATH = os.path.join("models", "autoencoder_model.h5")

# Ensure necessary directories exist.
if not os.path.exists("data"):
    print("Please create a 'data' folder and place 'creditcard.csv' inside it.")
    sys.exit(1)
if not os.path.exists("models"):
    os.makedirs("models")

# --- Block 2: Data Loading and Preprocessing ---
try:
    data = pd.read_csv(DATA_PATH)
except Exception as e:
    print("Error loading 'creditcard.csv'. Please ensure the file is in the 'data' folder.")
    print("Detailed error:", e)
    sys.exit(1)

print("Dataset loaded successfully. Shape:", data.shape)
# The dataset has 31 columns (Time, V1-V28, Amount, and Class).

# Scale the 'Time' and 'Amount' columns; PCA features (V1-V28) are assumed to be preprocessed.
scaler = StandardScaler()
data[['Time', 'Amount']] = scaler.fit_transform(data[['Time', 'Amount']])

# Separate features and label.
# We drop the 'Class' column from features; keep it aside for sample selection.
features = data.drop(['Class'], axis=1)

# --- Block 3: Train the Deep Autoencoder ---
# For anomaly detection using an autoencoder, we train the model only on normal transactions.
normal_data = features[data['Class'] == 0].values  # Convert to numpy array for training

# Determine the input dimension (should be 30: Time, V1-V28, Amount)
input_dim = normal_data.shape[1]
print("Input dimension:", input_dim)

# Build the autoencoder model using Keras Functional API.
# We create a simple autoencoder with a bottleneck structure.
input_layer = Input(shape=(input_dim,))
# Encoder
encoder = Dense(14, activation="relu")(input_layer)
encoder = Dense(7, activation="relu")(encoder)
# Bottleneck
bottleneck = Dense(3, activation="relu")(encoder)
# Decoder
decoder = Dense(7, activation="relu")(bottleneck)
decoder = Dense(14, activation="relu")(decoder)
output_layer = Dense(input_dim, activation="linear")(decoder)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

print("Training the autoencoder on normal transactions...")

# Train the autoencoder.
history = autoencoder.fit(normal_data, normal_data,
                          epochs=50,
                          batch_size=256,
                          shuffle=True,
                          validation_split=0.1,
                          verbose=1)

# (Optional) Save the trained model.
autoencoder.save(MODEL_PATH)
print("Autoencoder model trained and saved to:", MODEL_PATH)

# Compute reconstruction errors on the training set.
reconstructions = autoencoder.predict(normal_data)
train_errors = np.mean(np.square(normal_data - reconstructions), axis=1)
# Set the anomaly threshold as, for example, the 95th percentile of training errors.
threshold = np.percentile(train_errors, 95)
print("Reconstruction error threshold set to:", threshold)

# --- Block 4: Prepare Streaming Simulation Data ---
# To test the detection, we build a mixed sample that includes a few fraud cases.
fraud_data = features[data['Class'] == 1]
normal_sample = features[data['Class'] == 0].sample(n=95, random_state=42)
fraud_sample = fraud_data.sample(n=5, random_state=42) if len(fraud_data) >= 5 else fraud_data
streaming_data = pd.concat([fraud_sample, normal_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
print("Streaming simulation will use", len(streaming_data), "transactions.")

# --- Block 5: Define the Rule Engine ---
def rule_engine(reconstruction_error, threshold):
    """
    Interpret the reconstruction error.
    If the error exceeds the threshold, flag as an anomaly.
    """
    if reconstruction_error > threshold:
        return "ALERT: Suspicious Transaction Detected! (Blocked)"
    else:
        return "Transaction is normal."

# --- Block 6: Simulation Function ---
def run_simulation():
    """
    Simulate streaming data by processing each transaction one by one.
    For each transaction:
      - Compute the reconstruction error using the autoencoder.
      - Compare the error with the threshold via the rule engine.
      - Update the GUI's text area with the result.
      - Wait for 0.5 seconds to simulate a streaming delay.
    """
    for index, row in streaming_data.iterrows():
        # Reshape the row into a 2D array for prediction.
        features_row = row.values.reshape(1, -1)
        try:
            reconstruction = autoencoder.predict(features_row)
            # Compute the mean squared error for this sample.
            error = np.mean(np.square(features_row - reconstruction))
        except Exception as e:
            text_area.insert(tk.END, f"Error in processing transaction {index}: {e}\n")
            continue

        result = rule_engine(error, threshold)
        text_area.insert(tk.END, f"Transaction {index}: Reconstruction Error = {error:.4f} -> {result}\n")
        text_area.see(tk.END)
        time.sleep(0.5)

# --- Block 7: Build the Desktop GUI Using Tkinter ---
# Create the main application window.
root = tk.Tk()
root.title("Deep Learning Fraud Detection")
root.geometry("800x600")  # Window size: 800 x 600

# Create a ScrolledText widget to display log messages.
text_area = ScrolledText(root, wrap=tk.WORD, width=100, height=30)
text_area.pack(pady=20)

# Create a button to start the simulation. When clicked, it runs the simulation in a separate thread.
start_button = tk.Button(root, 
                         text="Start Fraud Detection Simulation", 
                         font=("Arial", 14),
                         command=lambda: threading.Thread(target=run_simulation, daemon=True).start())
start_button.pack(pady=10)

# Start the Tkinter event loop.
root.mainloop()
