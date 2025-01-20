import pandas 
from sklearn.model_selection import train_test_split

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