import pandas 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

dataframe = pandas.read_csv("delaney_solubility_with_descriptors.csv")

# Separate data as y
y = dataframe["logS"]
print("--------------------------")
print("y Data")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(y)
print("--------------------------")

# Separate data as x
x = dataframe.drop("logS", axis=1)
print("x Data")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(x)
print("--------------------------")

# Split data into training set 80% and test set 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

print("x Test")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(x_test)
print("--------------------------")
print("y Test")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(y_test)
print("--------------------------")

# Apply the model to make predictions using linear regression
lr = LinearRegression()
lr.fit(x_train, y_train) 

y_train_lr_prediction = lr.predict(x_train)
y_test_lr_prediction = lr.predict(x_test)

print("LR Prediction (Train)")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(y_train_lr_prediction)# Predict 80% of data
print("--------------------------")
print("LR Prediction (Test)")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(y_test_lr_prediction) # Predict remaining 20% of data
print("--------------------------")

# Evaluate the model's performance using linear regression
lr_train_mse = mean_squared_error(y_train, y_train_lr_prediction)
lr_train_r2 = r2_score(y_train, y_train_lr_prediction)

lr_test_mse = mean_squared_error(y_test, y_test_lr_prediction)
lr_test_r2 = r2_score(y_test, y_test_lr_prediction)

# Evaluate mean squared error and r2 score for lr in tabular form
lr_results = pandas.DataFrame(["Linear Regression", lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ["Method", "Training MSE", "Training R2", "Test MSE", "Test R2"]

print(lr_results)
print("--------------------------")

# Apply the model to make predictions using random forest
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)

y_train_rf_prediction = rf.predict(x_train)
y_test_rf_prediction = rf.predict(x_test)

print("RF Prediction (Train)")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(y_train_rf_prediction)# Predict 80% of data
print("--------------------------")
print("RF Prediction (Test)")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(y_test_rf_prediction) # Predict remaining 20% of data
print("--------------------------")

# Evaluate the model's performance using random forest
rf_train_mse = mean_squared_error(y_train, y_train_rf_prediction)
rf_train_r2 = r2_score(y_train, y_train_rf_prediction)

rf_test_mse = mean_squared_error(y_test, y_test_rf_prediction)
rf_test_r2 = r2_score(y_test, y_test_rf_prediction)

# Evaluate mean squared error and r2 score for rf in tabular form
rf_results = pandas.DataFrame(["Random Forest", rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ["Method", "Training MSE", "Training R2", "Test MSE", "Test R2"]

print(rf_results)
print("--------------------------")
