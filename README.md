# EX.NO-7-FEATURE-SELECTION
# AIM:
  To Perform the various feature selection techniques on a dataset and save the data to a file.

# EXPLANATION:
  Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.

# ALGORITHM:

## STEP-1:
  Read the given Data.
## STEP-2:
  Clean the Data Set using Data Cleaning Process.
## STEP-3:
  Apply Feature selection techniques to all the features of the data set.
## STEP-4:
  Save the data to the file.

# CODE:
## TITANIC DATASET:
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from google.colab import files

uploaded = files.upload()

df = pd.read_csv("titanic_dataset.csv")

df

df.isnull().sum()

df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])

df['Embarked'] = le.fit_transform(df['Embarked'].astype(str))

imputer = SimpleImputer(missing_values=np.nan, strategy='median')

df[['Age']] = imputer.fit_transform(df[['Age']])

print("Feature selection")

X = df.iloc[:, :-1]

y = df.iloc[:, -1]

selector = SelectKBest(chi2, k=3)

X_new = selector.fit_transform(X, y)

print(X_new)

df_new = pd.DataFrame(X_new, columns=['Pclass', 'Age', 'Fare'])

df_new['Survived'] = y.values

df_new.to_csv('titanic_transformed.csv', index=False)

print(df_new)

## CARPRICE DATASET:
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.ensemble import ExtraTreesRegressor

from google.colab import files

uploaded = files.upload()

df = pd.read_csv("CarPrice.csv")

df

df = df.drop(['car_ID', 'CarName'], axis=1)

le = LabelEncoder()

df['fueltype'] = le.fit_transform(df['fueltype'])

df['aspiration'] = le.fit_transform(df['aspiration'])

df['doornumber'] = le.fit_transform(df['doornumber'])

df['carbody'] = le.fit_transform(df['carbody'])

df['drivewheel'] = le.fit_transform(df['drivewheel'])

df['enginelocation'] = le.fit_transform(df['enginelocation'])

df['enginetype'] = le.fit_transform(df['enginetype'])

df['cylindernumber'] = le.fit_transform(df['cylindernumber'])

df['fuelsystem'] = le.fit_transform(df['fuelsystem'])

X = df.iloc[:, :-1]

y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print("Univariate Selection")

selector = SelectKBest(score_func=f_regression, k=10)

X_train_new = selector.fit_transform(X_train, y_train)

mask = selector.get_support()

selected_features = X_train.columns[mask]

model = ExtraTreesRegressor()

model.fit(X_train, y_train)

importance = model.feature_importances_

indices = np.argsort(importance)[::-1]

selected_features = X_train.columns[indices][:10]

df_new = pd.concat([X_train[selected_features], y_train], axis=1)

df_new.to_csv('CarPrice_new.csv', index=False)

print(df_new)

# OUTPUT:
## TITANIC DATASET:
![Screenshot (50)](https://user-images.githubusercontent.com/128498431/236686971-f2abe94c-c869-4d80-9dee-c8692b5a2c98.png)

![Screenshot (51)](https://user-images.githubusercontent.com/128498431/236687102-2b795f20-5f85-4663-8571-f4e56800a489.png)

![Screenshot (52)](https://user-images.githubusercontent.com/128498431/236687118-745319c6-d018-4133-babc-24b4675e72dd.png)

![Screenshot (53)](https://user-images.githubusercontent.com/128498431/236687134-bac2a613-e9bc-442b-8f94-c512151d973e.png)

## CARPRICE DATASET:
![Screenshot (54)](https://user-images.githubusercontent.com/128498431/236687149-504671b2-bf41-4920-b5e7-d2341a32faaf.png)

![Screenshot (55)](https://user-images.githubusercontent.com/128498431/236687177-910050b1-e1b4-4fd7-a5cc-629791701940.png)

# RESULT:
  Thus the various feature selection techniques was performed on the given datasets and output was verified successfully.






