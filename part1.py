import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ssl

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Load the dataset from the hosted URL
file_url = 'https://raw.githubusercontent.com/TamerAlaeddin/CS4375-Assignment1/main/data/auto-mpg.data'
column_names = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]
data = pd.read_csv(file_url, names=column_names, sep='\\s+')

# Pre-process the dataset
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)
data['origin'] = data['origin'].astype('category').cat.codes
data.drop(columns=['car name'], inplace=True)

# Convert all columns to numeric types
data = data.apply(pd.to_numeric)

# Print the dataset to check for any NaN or inf values
print(data.describe())

# Split the dataset
X = data.drop(columns=['mpg'])
y = data['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize parameters
def initialize_parameters(n_features):
    weights = np.zeros(n_features)
    bias = 0
    return weights, bias

# Define cost function
def compute_cost(X, y, weights, bias):
    n = len(y)
    predictions = X.dot(weights) + bias
    cost = (1/(2*n)) * np.sum((predictions - y)**2)
    return cost

# Implement gradient descent
def gradient_descent(X, y, weights, bias, learning_rate, iterations):
    n = len(y)
    cost_history = []
    
    for i in range(iterations):
        predictions = X.dot(weights) + bias
        errors = predictions - y
        
        weights_gradient = (1/n) * X.T.dot(errors)
        bias_gradient = (1/n) * np.sum(errors)
        
        weights -= learning_rate * weights_gradient
        bias -= learning_rate * bias_gradient
        
        cost = compute_cost(X, y, weights, bias)
        cost_history.append(cost)
        
        # Print cost to debug
        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost}")
    
    return weights, bias, cost_history

# Train the model
n_features = X_train.shape[1]
weights, bias = initialize_parameters(n_features)
learning_rate = 0.001  # Reduced learning rate
iterations = 1000

weights, bias, cost_history = gradient_descent(X_train, y_train, weights, bias, learning_rate, iterations)

# Evaluate the model
def evaluate_model(X, y, weights, bias):
    predictions = X.dot(weights) + bias
    mse = np.mean((predictions - y)**2)
    return mse

test_mse = evaluate_model(X_test, y_test, weights, bias)
print("Test MSE:", test_mse)

# Plot the cost function
plt.plot(range(iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function during Gradient Descent')
plt.show()

# Log the trials
with open('log.txt', 'w') as f:
    for i in range(iterations):
        f.write(f"Iteration {i+1}: Cost {cost_history[i]}\n")
