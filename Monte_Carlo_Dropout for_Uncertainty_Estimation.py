import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
X = np.linspace(-10, 10, 100).reshape(-1, 1)
y = np.sin(X) + 0.1 * np.random.randn(100, 1)

# Define a neural network with dropout layers
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),  # Dropout layer
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),  # Dropout layer
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# Train the model
model = create_model()
model.fit(X, y, epochs=200, batch_size=32, verbose=0)

# Perform Monte Carlo Dropout by making multiple predictions
def mc_dropout_predictions(X, model, num_samples=100):
    preds = [model(X, training=True) for _ in range(num_samples)]
    preds = tf.stack(preds, axis=0)
    return preds

# Get multiple predictions
preds = mc_dropout_predictions(X, model, num_samples=100)
y_pred_mean = tf.reduce_mean(preds, axis=0).numpy().flatten()
y_pred_std = tf.math.reduce_std(preds, axis=0).numpy().flatten()

# Plot the mean and uncertainty
plt.figure(figsize=(10, 6))
plt.plot(X, y, "b.", label="Observations")
plt.plot(X, y_pred_mean, "r-", label="Predictive mean")
plt.fill_between(X.flatten(),
                 y_pred_mean - 2 * y_pred_std,
                 y_pred_mean + 2 * y_pred_std,
                 color="red", alpha=0.2, label="Predictive uncertainty")
plt.legend()
plt.title("Monte Carlo Dropout Prediction with Uncertainty")
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()

