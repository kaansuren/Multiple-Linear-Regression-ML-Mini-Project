import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("tenis.csv")
print(veriler)

c = veriler.iloc[:,:1]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
veriler2 = veriler.apply(LabelEncoder().fit_transform)



ohe = OneHotEncoder()
c=ohe.fit_transform(c).toarray()

weather = pd.DataFrame(data=c, index=range(14), columns=["o","r","s"])
df = pd.concat([weather, veriler.iloc[:,1:3]], axis=1)
df = pd.concat([veriler2.iloc[:,-2:],df], axis=1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1:], test_size = 0.33, random_state=0)


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values=df.iloc[:,:-1], axis=1)
X_l = df.iloc[:,[0,1,2,3,4,5]].values

r= sm.OLS(endog=df.iloc[:,-1:], exog=X_l).fit()
print(r.summary())

df= df.iloc[:,1:]


X = np.append(arr = np.ones((14,1)).astype(int), values=df.iloc[:,:-1], axis=1)
X_l = df.iloc[:,[0,1,2,3,4]].values

r= sm.OLS(endog=df.iloc[:,-1:], exog=X_l).fit()
print(r.summary())



x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

lr.fit(x_train,y_train)

y_pred2 = lr.predict(x_test)
