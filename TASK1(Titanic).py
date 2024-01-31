Use the Titanic dataset to build a model that predicts whether a passenger on the Titanic survived or not. This is a classic beginner project with readily available data.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/content/Titanic-Dataset.csv")
df.head()

df.describe()

df.isnull().sum()

df.duplicated().sum()

plt.figure(figsize = (12,10))
sns.countplot(x = 'Survived', data = df)
plt.show()

sns.boxplot(df.Survived)
sns.distplot(df.Survived)

sns.boxplot(df.Age)
sns.distplot(df.Age)

df['Age'].fillna(df['Age'].median(), inplace = True)

df.isnull().sum()

df["Cabin"] = df["Cabin"].fillna('U')

df.head()

df["Cabin"].mode()

df["Embarked"].mode()

df.isnull().sum()

df['Survived'].value_counts()

sns.countplot(x = df['Survived'], hue = df['Pclass'])

plt.figure(figsize = (12,12))
for i,col in enumerate(['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']):
  plt.subplot(4,3, i+1)
  sns.boxplot(x = col, data = df)
plt.show()

sns.pairplot(df, hue = 'Survived')
plt.show()

sns.heatmap(df.corr(), cmap = "YlGnBu")
plt.show()

sns.countplot(x = df['Sex'], hue = df['Survived'])

sns.countplot(x = df['Age'], hue = df['Survived'])

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2)
for train_data, test_data in split.split(df, df[["Survived", "Pclass", "Sex"]]):
  strat_train_set = df.loc[train_data]
  strat_test_set = df.loc[test_data]

plt.subplot(1,2,1)
strat_train_set['Survived'].hist()
strat_train_set['Pclass'].hist()

plt.subplot(1,2,2)
strat_test_set['Survived'].hist()
strat_test_set['Pclass'].hist()

from sklearn.preprocessing import StandardScaler
sc_X.fit_transform(df.drop(["Survived"],axis = 1).select_dtypes(include=np.number))
X = pd.DataFrame(sc_X.fit_transform(df.drop(["Survived", "Name", "Sex", "Ticket", "Cabin", "Embarked"], axis=1)), columns = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'])
df["Sex"] = df["Sex"].astype("category").cat.codes
df["Ticket"] = df["Ticket"].astype("category").cat.codes
df["Cabin"] = df["Cabin"].astype("category").cat.codes
df["Embarked"] = df["Embarked"].astype("category").cat.codes

df.head()

y = df['Survived']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
test_scores = []
train_scores = []
for i in range(1,15):
  knn = KNeighborsClassifier(i)
  knn.fit(X_train, y_train)

  train_scores.append(knn.score(X_train, y_train))
  test_scores.append(knn.score(X_test, y_test))

max_train_score = max(train_scores)
train_scores_index = [i for i, v in enumerate(train_scores) if v == max_train_score]
print("Max Train Score {} % and k = {}".format(max_train_score*100, list(map(lambda x: x+1, train_scores_index))))

max_test_score = max(test_scores)
test_scores_index = [i for i, v in enumerate(test_scores) if v == max_test_score]
print("Max Test Score {} % and k = {}".format(max_test_score*100, list(map(lambda x: x+1, test_scores_index))))

plt.figure(figsize = (12,6))
p = sns.lineplot(x = range(1,15), y = train_scores, marker = '*', label = 'Train Score')
p = sns.lineplot(x = range(1,15), y = test_scores, marker = 'o', label = 'Test Score')

knn = KNeighborsClassifier(14)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

from sklearn.metrics import confusion_matrix
y_pred = knn.predict(X_test)
confusion_matrix(y_test, y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

