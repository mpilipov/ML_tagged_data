#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge, LassoCV
from sklearn.metrics import mean_squared_error

pd.options.display.max_columns = 20000
pd.options.display.max_rows = 20000
pd.options.display.expand_frame_repr = False

df = pd.read_csv("bikes_rent.csv")
# перемешивание элементов выборки
df_shuffled = shuffle(df, random_state=123)
# все столбцы в диапазоне перемешанной выборки кроме последнего
X = scale(df_shuffled[df_shuffled.columns[:-1]])
# последний столбец cnt
y = df_shuffled["cnt"]
alphas = np.arange(1, 500, 50)

coefs_lasso = np.zeros((alphas.shape[0], X.shape[1]))  # матрица весов размера (число регрессоров) x (число признаков)
coefs_ridge = np.zeros((alphas.shape[0], X.shape[1]))

#веса регуляризатора L1
for i in range(len(alphas)):
    lass0_model = Lasso(alpha=alphas[i])
    lass0_model.fit(X, y)
    for j in range(X.shape[1]):
        coefs_lasso[i][j] = lass0_model.coef_[j]

for i in range(10):
    for j in range(12):
        print(coefs_lasso[i][j], end=' ')
    print()

#веса регуляризатора L2
for i in range(len(alphas)):
    ridge_model = Ridge(alpha=alphas[i])
    ridge_model.fit(X, y)
    for j in range(X.shape[1]):
        coefs_ridge[i][j] = ridge_model.coef_[j]

for i in range(10):
    for j in range(12):
        print(coefs_ridge[i][j], end=' ')
    print()

# построение графиков

plt.figure(figsize=(8, 5))
for coef, feature in zip(coefs_lasso.T, df.columns):
    plt.plot(alphas, coef, label=feature, color=np.random.rand(3))
plt.legend(loc="upper left", bbox_to_anchor=(0.9, 1.1))
plt.xlabel("alpha")
plt.ylabel("feature weight")
plt.title("Lasso")

plt.figure(figsize=(8, 5))
for coef, feature in zip(coefs_ridge.T, df.columns):
    plt.plot(alphas, coef, label=feature, color=np.random.rand(3))
plt.legend(loc="upper left", bbox_to_anchor=(0.9, 1.1))
plt.xlabel("alpha")
plt.ylabel("feature weight")
plt.title("Ridge")

plt.show()

# кросс-валидация
alphas = np.arange(1, 100, 5)
mse_alphas=np.zeros(10)
reg = LassoCV(alphas=alphas).fit(X, y)
mse_mean = reg.mse_path_.mean(axis=1)
plt.plot(alphas, mse_mean)

# лучшая альфа, выбранная на кросс - валидации
print('alpha=', reg.alpha_)
learn_coeffs = reg.coef_

# итоговые коэффициенты, основанные на лучшем альфа, которую выбрали на кросс-валидации
print(reg.coef_)

# вывод пар признак-коэффициент
for i in range (0, 12):
    print(df_shuffled.columns[i], " - ", learn_coeffs[i] )

print("alphas:")
#print(reg.mse_path_)
plt.xlabel("alpha")
plt.ylabel("mse")
plt.show()
col1 = [a[0] for a in reg.mse_path_]
col2 = [a[1] for a in reg.mse_path_]
col3 = [a[2] for a in reg.mse_path_]
print(col1, "\n", col2,"\n", col3)
print('лучшие альфа на каждом разбиении')
print(reg.alphas_[col1.index(min(col1))])
print(reg.alphas_[col2.index(min(col2))])
print(reg.alphas_[col3.index(min(col3))])

plt.plot(reg.alphas_, col1)
plt.plot(reg.alphas_, col2)
plt.plot(reg.alphas_, col3)
plt.show()