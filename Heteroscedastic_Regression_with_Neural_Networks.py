import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data with varying noise levels
X = np.linspace(-10, 10, 100).reshape(-1, 1)
y = np.sin(X) + (0.1 + 0.5 * np.abs(X / 10)) * np.random.randn(100, 1)  # Increasing noise

# Define a neural network for heteroscedastic regression
inputs = tf.keras.layers.Input(shape=(1,))
x = tf.keras.layers.Dense(64, activation="relu")(inputs)
x = tf.keras.layers.Dense(64, activation="relu")(x)
mean_output = tf.keras.layers.Dense(1, name="mean")(x)
log_stddev_output = tf.keras.layers.Dense(1, name="log_stddev")(x)  # Log of stddev for stability
model = tf.keras.Model(inputs=inputs, outputs=[mean_output, log_stddev_output])

# Custom loss function: Heteroscedastic Negative Log-Likelihood
def heteroscedastic_nll(y_true, y_pred):
    mean_pred, log_stddev_pred = y_pred
    stddev_pred = tf.exp(log_stddev_pred)
    return tf.reduce_mean(0.5 * tf.math.log(2 * np.pi * tf.square(stddev_pred)) +
                          tf.square(y_true - mean_pred) / (2 * tf.square(stddev_pred)))

# Compile and train the model
model.compile(optimizer="adam", loss=heteroscedastic_nll)
model.fit(X, [y, y], epochs=200, batch_size=32, verbose=0)

# Make predictions
y_pred_mean, y_pred_log_stddev = model.predict(X)
y_pred_stddev = np.exp(y_pred_log_stddev).flatten()

# Plot the mean and varying uncertainty
plt.figure(figsize=(10, 6))
plt.plot(X, y, "b.", label="Observations")
plt.plot(X, y_pred_mean, "r-", label="Predictive mean")
plt.fill_between(X.flatten(),
                 y_pred_mean.flatten() - 2 * y_pred_stddev,
                 y_pred_mean.flatten() + 2 * y_pred_stddev,
                 color="red", alpha=0.2, label="Predictive uncertainty")
plt.legend()
plt.title("Heteroscedastic Regression with Varying Uncertainty")
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()
