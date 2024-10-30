import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.datasets import load_iris

# Load the Iris dataset
data = load_iris()
X = data.data  # Feature data (150, 4)
y = data.target  # Class labels

# Initialize the SOM with a 10x10 grid and feature length of 4 (Iris dataset)
som = MiniSom(x=10, y=10, input_len=4, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)

# Train the SOM
som.train_random(X, num_iteration=100)

# Plot the SOM with class labels
plt.figure(figsize=(8, 8))
for i, x in enumerate(X):
    winner = som.winner(x)  # Get the winning neuron for each data point
    plt.text(winner[0] + 0.5, winner[1] + 0.5, str(y[i]), color='red' if y[i] == 0 else 'blue' if y[i] == 1 else 'green')
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title("Self-Organizing Map for Iris Dataset")
plt.show()

