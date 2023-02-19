import pandas as pd
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
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
        X_train, X_test = train_data.iloc[train_index, :-1], train_data.iloc[test_index, :-1]
        y_train, y_test = X_train["Close"], X_test["Close"]
        model.fit(X_train[['Date']], y_train)
        y_pred = model.predict(X_test[['Date']])
        mse = mean_squared_error(y_test, y_pred)
        mse_list.append(mse)
    avg_mse = np.mean(mse_list)
    if avg_mse < best_mse:
        best_mse = avg_mse
        best_model = model
        best_model_name = model_names[i]
        
# Train the best model on the entire training data and make predictions
best_model.fit(train_data[['Date']], train_data["Close"])
y_pred = best_model.predict(test_data[['Date']])
mse = mean_squared_error(test_data["Close"], y_pred)


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
print("Predicted stock price for the next month: ", next_month_price)
next_price = best_model.predict(np.array([[next_month]]))
print("Predicted price for next month: ", next_price[0])

plt.plot(data["Date"], data["Close"], label="Actual Price")
plt.plot(np.array([next_month]), next_month_price, 'ro', label="Predicted Price")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.show()




