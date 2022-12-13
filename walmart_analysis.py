import pandas as pd
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from pmdarima import auto_arima
import sklearn.metrics

# load walmart dataset
url = "https://raw.githubusercontent.com/wahub-ahmed/walmart-sales-prediction/main/train.csv"
data = pd.read_csv(url)

# convert the 'Date' column to datetime format and add 'Month' and 'Day' columns
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# sort the data by the 'Date' column
data = data.sort_values(by = 'Date')

# set the 'Date' column as the index of the DataFrame
data = data.set_index('Date')

# set a cutoff date to split the data into training and testing sets
date_cutoff = datetime.datetime(2012, 4, 13)

# containing the sum sales of each date
actual = data.groupby("Date").sum()

# making baseline prediction of using previous data sales
actual["baseline"] = actual['Weekly_Sales'].shift(1)

# getting the baseline for the dates needed
baseline = pd.Series(actual["baseline"][actual.index >= date_cutoff])

# split the data into features and target
X = data.drop(columns = ['Weekly_Sales'])
y = data['Weekly_Sales']

# split the data into training and testing sets for Decision Tree
X_train = X[X.index < date_cutoff]
X_test = X[X.index >= date_cutoff]
y_train = y[y.index < date_cutoff]
y_test = y[y.index >= date_cutoff]

# train a decision tree on the training data
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

# make predictions on the testing data
dt_preds = dt.predict(X_test)

# give proper index to the predictions
dt_series = pd.Series(dt_preds).set_axis(y_test.index)

# sum the prediction for each date
tree = dt_series.groupby("Date").sum()

# split data into training and testing for ARIMA

train = actual[actual.index < date_cutoff]['Weekly_Sales']
test = actual[actual.index >= date_cutoff]['Weekly_Sales']

# train ARIMA model with training data
model = auto_arima(y = train, m = 52, seasonal = True, stepwise=True, approximation=True)

# make prediction for the length of test data
arima = pd.Series(model.predict(n_periods = len(test))).set_axis(test.index)

# ploting Baseline Predictions
actual["Weekly_Sales"].plot(legend = True, label = "Actual")
baseline.plot(legend=True, label = "Baseline")
plt.title('Baseline Prediction of Walmart Sales')
plt.ylabel('Weekly Sales')
plt.show()

# plotting Decision Tree Predictions
actual["Weekly_Sales"].plot(legend = True, label = "Actual")
tree.plot(legend=True, label = "Decision Tree")
plt.title('Decision Tree Prediction of Walmart Sales')
plt.ylabel('Weekly Sales')
plt.show()

# plotting ARIMA Predictions
actual["Weekly_Sales"].plot(legend = True, label = "Actual")
arima.plot(legend=True, label = "Baseline")
plt.title('ARIMA Prediction of Walmart Sales')
plt.ylabel('Weekly Sales')
plt.show()
