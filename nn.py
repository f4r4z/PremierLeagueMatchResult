import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import glob

# file to data frame and concatenation
all_files = glob.glob('./data' + '/*.csv')
df = pd.concat(map(pd.read_csv, all_files), sort=True)

# encode strings to int
enc = LabelEncoder()
enc.fit(df['FTR'])
df['FTR'] = enc.transform(df['FTR'])
enc.fit(df['HTR'])
df['HTR'] = enc.transform(df['HTR'])

# x and y
x = df[['HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'B365H', 'B365D', 'B365A']]
y = df['FTR']


# divide data set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33)

# neural network: stochastic gradient, step size, layers and neurons, randomized state
clf = MLPClassifier(solver='lbfgs', alpha=0.0000001,  hidden_layer_sizes=(10, 10))  #previous: random_state=1

# fit
clf.fit(x_train, y_train)

# predict test data
yhat = clf.predict(x_test)

# accuracy calculation
score = clf.score(x_test, y_test)
print(score)

# print(list.tolist())
# print(yhat.tolist())

# original_headers = list(df.columns.values)

