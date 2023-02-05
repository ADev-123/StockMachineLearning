import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# Load the stock data
data = pd.read_csv("tesla_stock_data.csv")

# Preprocess the data
data = data.dropna()

# Split the data into training and testing sets
train_data = data[:int(0.8*len(data))]
test_data = data[int(0.8*len(data)):]

# Choose a model
model1 = LinearRegression()
model2 = SVR()
model3 = RandomForestRegressor()

# Train the model and evaluate the performance
tscv = TimeSeriesSplit(n_splits=5)
models = [model1, model2, model3]
model_names = ["Linear Regression", "Support Vector Regression", "Random Forest Regression"]
best_mse = float("inf")
best_model = None
best_model_name = ""
for i, model in enumerate(models):
    mse_list = []
    for train_index, test_index in tscv.split(train_data):
        X_train, X_test = train_data.iloc[train_index], train_data.iloc[test_index]
        y_train, y_test = X_train["Close"], X_test["Close"]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_list.append(mse)
    avg_mse = np.mean(mse_list)
    if avg_mse < best_mse:
        best_mse = avg_mse
        best_model = model
        best_model_name = model_names[i]
        
# Train the best model on the entire training data and make predictions
best_model.fit(train_data, train_data["Close"])
y_pred = best_model.predict(test_data)
mse = mean_squared_error(test_data["Close"], y_pred)
print("Mean Squared Error: ", mse)
print("Best Model: ", best_model_name)
