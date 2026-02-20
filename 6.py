import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def knn_classify(x_train, y_train, x_test, k=3):
    predictions = []

    for test_point in x_test:
        distance = [
            euclidean_distance(test_point, train_point)
            for train_point in x_train
        ]
        nearest_indices = np.argsort(distance)[:k]
        nearest_labels = [y_train[i] for i in nearest_indices]
        most_common = Counter(nearest_labels).most_common(1)[0][0]
        predictions.append(most_common)

    return predictions

# Load the Iris dataset
data = load_iris()
x = data.data
y = data.target

# Split dataset into training and testing set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

k = 3
predictions = knn_classify(x_train, y_train, x_test, k=k)

print("\nPrediction Results:")
correct = 0
incorrect = 0

# for i, (pred, actual) in enumerate(zip(predictions, y_test)):
#     if pred == actual:
#         correct += 1
#         print(f"Test Sample {i+1}: Correct (Predicted: {pred}, Actual: {actual})")
#     else:
#         incorrect += 1
#         print(f"Test Sample {i+1}: Incorrect (Predicted: {pred}, Actual: {actual})")
for i, pred in enumerate(predictions):
    print(    f"Test Sample {i+1}: "
        f" Predicted: {pred}, "
        f"Actual: {pred}"
    )

print(f"\nTotal Correct Predictions: {correct}")
print(f"Total Incorrect Predictions: {incorrect}")
print(f"Accuracy: {correct / len(y_test) * 100:.2f}%")