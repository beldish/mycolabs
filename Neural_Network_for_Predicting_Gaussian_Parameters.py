import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError

# Generate sample data (y ~ N(μ, σ^2))
np.random.seed(0)
X = np.random.rand(1000, 1)  # Random input feature
y_true_mean = 2 * X.squeeze()  # Mean of Gaussian (true model)
y_true_stddev = 0.5  # Fixed stddev
y = y_true_mean + y_true_stddev * np.random.randn(1000)  # Simulated observations

# Define neural network
inputs = Input(shape=(1,))
mean_output = Dense(1, name="mean")(inputs)
stddev_output = Dense(1, activation="softplus", name="stddev")(inputs)  # Ensure positive stddev

model = Model(inputs=inputs, outputs=[mean_output, stddev_output])

# Custom loss for Gaussian log-likelihood
def gaussian_nll(y_true, mean_pred, stddev_pred):
    return tf.reduce_mean(0.5 * tf.math.log(2 * np.pi * tf.square(stddev_pred)) +
                          tf.square(y_true - mean_pred) / (2 * tf.square(stddev_pred)))

# Compile model with custom loss
model.compile(optimizer="adam", loss=lambda y, pred: gaussian_nll(y, pred[0], pred[1]))

# Train model
model.fit(X, [y, y], epochs=100, batch_size=32, verbose=1)

# Predict mean and stddev
predicted_mean, predicted_stddev = model.predict(X)

# Example predictions
print("Predicted mean:", predicted_mean[:5].flatten())
print("Predicted stddev:", predicted_stddev[:5].flatten())

