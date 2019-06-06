import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier



data = pd.read_csv("train.csv").as_matrix()
clf = DecisionTreeClassifier()     #object is created

#training
xtrain = data[0:21000, 1:]         #0:21000 means entire dataset and 1: means consider whatever is after 1 column
train_label = data[0:21000, 0]     #0:21000 means entire dataset and 0 means consider only the column 0 which contains the label digit

clf.fit(xtrain, train_label)

#testing
xtest = data[21000: , 1: ]
actual_label=data[21000:, 0]

d=xtest[8]
d.shape=(28,28)
pt.imshow(255-d, cmap='gray')

print(clf.predict( [xtest[8]] ))

pt.show()
