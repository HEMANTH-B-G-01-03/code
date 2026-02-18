import numpy as np
def sigmoid(x):
 return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
 return x * (1 - x)
# Define input data and expected output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]]) # XOR output
np.random.seed(42)
# Set the architecture of the neural network
input_layer_neurons = X.shape[1] # 2 inputs
hidden_layer_neurons = 4
output_neurons = 1
# Initialize weights and biases
weights_input_hidden = np.random.uniform(size=(input_layer_neurons,
hidden_layer_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_layer_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))
learning_rate = 0.5
# Training the neural network
for epoch in range(10000):
 # Forward pass
 hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
 hidden_layer_output = sigmoid(hidden_layer_input)
 output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
 predicted_output = sigmoid(output_layer_input)

 error = y - predicted_output

 # Backpropagation
 d_predicted_output = error * sigmoid_derivative(predicted_output)
 error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
 d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
 # Update weights and biases
 weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
 bias_output += np.sum(d_predicted_output, axis=0, keepdims=True)*learning_rate
 weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
 bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
# Display the trained output
print("Trained Output:")
print(predicted_output)
# Testing the trained neural network
print("\nTesting the ANN:")
for i, test_input in enumerate(X):
 hidden_layer_input = np.dot(test_input, weights_input_hidden) + bias_hidden
 hidden_layer_output = sigmoid(hidden_layer_input)
 output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
 predicted_output = sigmoid(output_layer_input)
 print(f"Input: {test_input}, Predicted Output: {predicted_output}")