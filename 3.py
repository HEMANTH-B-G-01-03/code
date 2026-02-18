#SVM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
# Generate a synthetic dataset for binary classification
x, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.5)
# Train a support vector machine model
model = SVC(kernel='linear')
model.fit(x, y)
# Define a function to plot the decision boundary
def plot_decision_boundary(x, y, model):
 x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
 y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
 xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
 np.arange(y_min, y_max, 0.01))
 z = model.predict(np.c_[xx.ravel(), yy.ravel()])
 z = z.reshape(xx.shape)
 plt.contourf(xx, yy, z, alpha=0.8, cmap=plt.cm.coolwarm)
 plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
 plt.xlabel('Feature 1')
 plt.ylabel('Feature 2')
 plt.title('SVM decision boundary')
 plt.show() # Add parentheses here to display the plot
# Plot the decision boundary
plot_decision_boundary(x, y, model) 