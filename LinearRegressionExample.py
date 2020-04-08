import pandas as pd
import numpy as np
import sklearn
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#import the dataset using panda read_csv method
df = pd.read_csv('heart.csv')
tdf = pd.read_csv('test_heart.csv')

#convert the dataset to a numpy array and remove the target column
samples = np.array(df.drop(["target"],1))
test_samples = np.array(tdf.drop(["target"],1))
lables = np.array(df["target"])
test_lables = np.array(tdf["target"])
#transform the data to standerdized values
scaler = StandardScaler()
scaler.fit(samples)
samples = scaler.transform(samples)
scaler.fit(test_samples)
test_samples = scaler.transform(test_samples)
#split 20% of the dataset into two training and validation part using train_test_split method
x_train, x_test, y_train, y_test = train_test_split(samples, lables, test_size=0.1)

#for support vector machine
#model = svm.SVC()
#for linear regression algorithm
model = LinearRegression()
#train the model
model.fit(x_train, y_train)
#test the model
model.score(x_test, y_test)
#predict the new unseen examples using the model
model.predict(test_samples)
