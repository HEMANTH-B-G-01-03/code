import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = np.array([
    [1200, 200],
    [1500, 250],
    [1700, 200],
    [2100, 400],
    [2300, 450],
    [2500, 500]
])

x = data[:, 0].reshape(-1, 1)
y = data[:, 1]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

def knn_regression(x_train, y_train, x_test, k=3):
    predictions = []

    for test_point in x_test:
        distances = np.sqrt(np.sum((x_train - test_point) ** 2, axis=1))
        nearest_indices = np.argsort(distances)[:k]
        nearest_value = y_train[nearest_indices]
        prediction = np.mean(nearest_value)
        predictions.append(prediction)

    return np.array(predictions)

y_pred = knn_regression(x_train, y_train, x_test, k=3)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error: {mse:.2f}")

print("\nTest Predictions:")
for i, (size, actual, pred) in enumerate(zip(x_test.flatten(), y_test, y_pred)):
    print(
        f"House Size: {size}, "
        f"Actual price: {actual}, "
        f"Predicted price: {pred:.2f}"
    )