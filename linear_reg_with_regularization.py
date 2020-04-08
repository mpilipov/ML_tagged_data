import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
pd.options.display.expand_frame_repr = False

df = pd.read_csv("bikes_rent.csv")
print(df.head())

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
for i, f in enumerate(df.columns[:-1]):
    df.plot(f, "cnt", subplots=True, kind="scatter", ax=axes[i // 4, i % 4])
plt.show()

# корреляции всех признаков с cnt
print(df.corrwith(df['cnt']))
# корреляции всех признаков со всеми
print(df.mean())

# перемешивание элементов выборки
df_shuffled = shuffle(df, random_state=123)

# все столбцы в диапазоне перемешанной выборки кроме последнего
X = scale(df_shuffled[df_shuffled.columns[:-1]])

# последний столбец cnt
y = df_shuffled["cnt"]

# обучение линейной регрессии на данных
# создадим модель
model = LinearRegression()

#обучение
model.fit(X, y)

print(model.coef_)
for i in range(12):
    print(df.columns[i]," || ", model.coef_[i])

lasso_model = Lasso()
lasso_model.fit(X, y)
print(lasso_model.coef_)
for i in range(12):
    print(df.columns[i]," || ", lasso_model.coef_[i])

print("sdfsdfsdfssdfsdfsd-----------------------------------")

ridge_model = Ridge()
ridge_model.fit(X, y)
print(ridge_model.coef_)
for i in range(12):
    print(df.columns[i]," || ", ridge_model.coef_[i])
print(ridge_model.coef_.T)


