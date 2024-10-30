import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from pyGTM import GTM

# Step 1: Load Dataset (Iris dataset with 4 features, 3 classes)
data = load_iris()
X = data.data  # Features: shape (150, 4)
y = data.target  # Labels (for visualization)

# Step 2: Define the GTM model parameters
latent_dim = 2  # We want a 2D latent space for visualization
grid_shape = (10, 10)  # Defines a 10x10 grid in the latent space
rbf_width = 0.5  # Width of the radial basis functions
regularization = 0.001  # Regularization parameter

# Step 3: Initialize the GTM model
gtm = GTM(grid_shape=grid_shape, latent_dim=latent_dim, rbf_width=rbf_width, regularization=regularization)

# Step 4: Fit the GTM model to the data
gtm.fit(X)

# Step 5: Project the data to the 2D latent space
latent_points = gtm.transform(X)

# Step 6: Plot the 2D latent space with color-coded classes
plt.figure(figsize=(8, 6))
for i, label in enumerate(np.unique(y)):
    plt.scatter(latent_points[y == label, 0], latent_points[y == label, 1], label=data.target_names[label])
plt.title("Generative Topographic Mapping (GTM) on Iris Dataset")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.legend()
plt.show()
