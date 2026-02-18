import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class SimpleLinearRegression:
    def __init__(self):
        self.slope = 0
        self.intercept = 0

    def fit(self, x, y):
        # calculate slope and intercept using formula
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        num = np.sum((x - x_mean) * (y - y_mean))
        deno = np.sum((x - x_mean) ** 2)

        self.slope = num / deno
        self.intercept = y_mean - self.slope * x_mean

    def predict(self, x):
        # predict target variable
        return self.slope * x + self.intercept

    def evaluate(self, x, y):
        # evaluate model using mean squared error
        y_pred = self.predict(x)
        mse = np.mean((y - y_pred) ** 2)
        return mse


# generate a synthetic dataset for demo
np.random.seed(42)
x = 2 * np.random.rand(100, 1).flatten()
y = 4 + 3 * x + np.random.randn(100)

# split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# train and evaluate the model
model = SimpleLinearRegression()
model.fit(x_train, y_train)

print(f"Slope: {model.slope}")
print(f"Intercept: {model.intercept}")

mse = model.evaluate(x_test, y_test)
print(f"Mean Squared Error on Test Set: {mse}")

# plot the results
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x, model.predict(x), color='red', label='Linear Fit')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
