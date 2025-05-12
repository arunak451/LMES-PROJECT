import numpy as np

arr = np.array([1, 2, 3, 4])
print("Mean:", np.mean(arr))

import pandas as pd

data = {'Name': ['Alice', 'Bob'], 'Age': [25, 30]}
df = pd.DataFrame(data)
print(df.describe())

import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [4, 5, 6]
plt.plot(x, y)
plt.title("Simple Plot")
plt.show()

import seaborn as sns
import pandas as pd

df = sns.load_dataset("iris")
sns.pairplot(df, hue="species")

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print("Accuracy:", clf.score(X_test, y_test))

