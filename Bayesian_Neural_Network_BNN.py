import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

# Generate synthetic data
X = np.linspace(-10, 10, 100).reshape(-1, 1)
y = np.sin(X) + 0.1 * np.random.randn(100, 1)

# Define Bayesian Neural Network
tfd = tfp.distributions

model = tf.keras.Sequential([
    tfp.layers.DenseFlipout(64, activation="relu"),
    tfp.layers.DenseFlipout(64, activation="relu"),
    tfp.layers.DenseFlipout(1)
])

# Define the negative log-likelihood as loss
def nll(y, y_pred):
    return -y_pred.log_prob(y)

# Compile model
model.compile(optimizer="adam", loss=nll)

# Train the model
model.fit(X, y, epochs=200, batch_size=32, verbose=0)

# Make predictions
y_pred_dist = model(X)  # Predictive distribution
y_pred_mean = y_pred_dist.mean().numpy().flatten()
y_pred_std = y_pred_dist.stddev().numpy().flatten()

# Plot the mean and uncertainty (standard deviation)
plt.figure(figsize=(10, 6))
plt.plot(X, y, "b.", label="Observations")
plt.plot(X, y_pred_mean, "r-", label="Predictive mean")
plt.fill_between(X.flatten(),
                 y_pred_mean - 2 * y_pred_std,
                 y_pred_mean + 2 * y_pred_std,
                 color="red", alpha=0.2, label="Predictive uncertainty")
plt.legend()
plt.title("Bayesian Neural Network Prediction with Uncertainty")
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()
