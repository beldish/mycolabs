from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Generate sample data
X_train = np.random.rand(100, 1) * 10
y_train = 2 * np.sin(X_train).ravel() + np.random.normal(0, 0.5, X_train.shape[0])

# Define the Gaussian Process model with RBF kernel
kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Fit to data using maximum likelihood estimation
gp.fit(X_train, y_train)

# Make predictions with uncertainty
X_test = np.linspace(0, 10, 1000).reshape(-1, 1)
y_pred, sigma = gp.predict(X_test, return_std=True)

# Plot results
import matplotlib.pyplot as plt

plt.figure()
plt.plot(X_train, y_train, 'r.', markersize=10, label="Observations")
plt.plot(X_test, y_pred, 'b-', label="Prediction")
plt.fill_between(X_test.flatten(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, color="b", alpha=0.2)
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("Gaussian Process Regression with Confidence Interval")
plt.legend()
plt.show()
