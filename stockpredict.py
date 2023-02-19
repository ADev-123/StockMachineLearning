import pandas as pd
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error
from datetime import datetime


# Load the stock data
data = pd.read_csv("InsertName.csv")
# Preprocess the data
data = data.dropna()

# Convert the dates to integers
data['Date'] = data['Date'].apply(lambda x: (datetime.strptime(x, '%Y-%m-%d') - datetime(2010,1,1)).days)

# Split the data into training and testing sets
train_data = data[:int(0.8*len(data))]
test_data = data[int(0.8*len(data)):]

# Choose a model
models = [LinearRegression(), SVR(), RandomForestRegressor()]

# Perform K-fold cross-validation and hyperparameter tuning
best_mse = float("inf")
best_model = None
for model in models:
    param_grid = {}
    if isinstance(model, SVR):
        param_grid = {"C": [0.01, 0.1, 1, 10], "gamma": [0.01, 0.1, 1, 10]}
    elif isinstance(model, RandomForestRegressor):
        param_grid = {"n_estimators": [50, 100, 200, 500], "max_depth": [3, 5, 10, None]}
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)
    grid_search.fit(train_data[["Date"]], train_data["Close"])
    
    if -grid_search.best_score_ < best_mse:
        best_mse = -grid_search.best_score_
        best_model = grid_search.best_estimator_

# Train the best model on the entire training data and make predictions
best_model.fit(train_data[['Date']], train_data["Close"])
y_pred = best_model.predict(test_data[['Date']])
mse = mean_squared_error(test_data["Close"], y_pred)

# Print model evaluation metrics and predictions for next month
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print("Mean Squared Error: ", mse)
    print("Best Model: ", best_model)
    
last_date = data["Date"].iloc[-1]
last_date = pd.Timestamp(datetime(2010,1,1) + pd.Timedelta(days=last_date))
next_month = (last_date + pd.Timedelta("30D")).to_julian_date()
next_month = (next_month - datetime(2010,1,1).toordinal()) / 365 # convert to number of years

next_month_data = pd.DataFrame({"Date": [next_month]})

next_month_price = best_model.predict(np.array([next_month]).reshape(-1, 1))[0]
print("Predicted price for next month: ", next_month_price)

# Visualize the results
plt.plot(train_data["Date"], train_data["Close"], label="Training Data")
plt.plot(test_data["Date"], test_data["Close"], label="Testing Data")
plt.plot(np.array([next_month]), next_month_price, 'ro', label="Predicted Price")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.show()
