import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import ssl

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Load the dataset from GitHub
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

# Split the dataset
X = data.drop(columns=['mpg'])
y = data['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Use Scikit-learn's SGDRegressor
model = SGDRegressor(max_iter=1000, tol=1e-3)
model.fit(X_train, y_train)

# Evaluate the model
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_mse = np.mean((train_predictions - y_train)**2)
test_mse = np.mean((test_predictions - y_test)**2)
print("Train MSE:", train_mse)
print("Test MSE:", test_mse)
