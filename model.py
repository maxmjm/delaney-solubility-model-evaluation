import pandas 

dataframe = pandas.read_csv("delaney_solubility_with_descriptors.csv")

# Separate data as y
y = dataframe["logS"]
print(y)

# Separate data as x
x = dataframe.drop("logS", axis=1)
print(x)
