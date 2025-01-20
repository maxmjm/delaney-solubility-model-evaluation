import pandas 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataframe = pandas.read_csv("delaney_solubility_with_descriptors.csv")

# Separate data as y
y = dataframe["logS"]
print(y)

# Separate data as x
x = dataframe.drop("logS", axis=1)
print(x)

# Split data into training set 80% and test set 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

print(x_test)

# Apply the model to make predictions
lr = LinearRegression()
lr.fit(x_train, y_train) 

y_train_lr_prediction = lr.predict(x_train)
y_test_lr_prediction = lr.predict(x_test)

print(y_train_lr_prediction) # Predict 80% of data
print(y_test_lr_prediction) # Predict remaining 20% of data
