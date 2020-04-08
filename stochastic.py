import pandas as pd
import numpy as np
from statistics import median
from matplotlib import pyplot as plot

# сркв ошибка прогноза значений
from matplotlib.pyplot import xlabel, ylabel


def mercer(y, y_pred):
    # print(type(y), type(y_pred))
    sq = np.subtract(y, y_pred) ** 2
    # print(sq)
    return sq.mean()


def normal_equation(X, y):
    return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))


def linear_prediction(X, W):
    y_res = np.linspace(0, 0, 200)
    for i in range(len(X)):
        for j in range(0, len(W) - 1):
            y_res[i] = y_res[i] + X[i, j] * W[j]
        y_res[i] = y_res[i] + W[3]
    return y_res


def stochastic_gradient_step(X, y, w, train_ind, eta=0.01):
    X = np.array(X)
    y = np.array(y)
    w = np.array(w)
    l = len(y)
    print('xtri=',X[train_ind])
    print('w=',w)
    grad0 = (np.dot(X[train_ind], w) - y[train_ind]) * 2 * X[train_ind, 0]
    grad1 = (np.dot(X[train_ind], w) - y[train_ind]) * 2 * X[train_ind, 1]
    grad2 = (np.dot(X[train_ind], w) - y[train_ind]) * 2 * X[train_ind, 2]
    grad3 = (np.dot(X[train_ind], w) - y[train_ind]) * 2 * X[train_ind, 3]

    return w - eta / l * np.array([grad0, grad1, grad2, grad3])


def stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e4, min_weight_dist=1e-8, seed=42, verbose=False):
    w_dist = np.inf
    w = w_init
    errors = []
    iter_num = 0
    np.random.seed(seed)
    while w_dist > min_weight_dist and iter_num < max_iter:
        random_ind = np.random.randint(len(X))

        w_new = stochastic_gradient_step(X, y, w, random_ind, eta)
        w_dist = np.linalg.norm(w - w_new)

        errors.append(mercer(y, linear_prediction(X, w)))
        w = w_new
        iter_num = iter_num + 1
    return w, errors


adver_data = pd.read_csv('advertising1.csv', index_col='Index')
X = np.array(adver_data.values[:, 0:3])
y = np.array(adver_data.values[:, 3])
y2 = adver_data['Sales'].values
#print('yy2=', y - y2)
#print(pd.DataFrame.head(adver_data))

# масштабирование столбцов матрицы X
stds, means = np.std(X, axis=0), np.mean(X, axis=0)
X = (X - means) / stds

# добавление единичного столбца
ones = np.ones((200, 1))
X = np.hstack((X, np.ones((200, 1))))
#print('X=', X)

# расчёт сркв ошибки Sales при предсказании медианного значения по исходной выборке
median_y = np.array([median(y) for i in range(0, len(y))])
#print(median_y)
print('answer1=', round(mercer(y, median_y), 3))

# расчёт вектора весов w по нормальному уравнению линрег
W = normal_equation(X, y)
#print(W)

# предсказание продажи
Y = 0
for i in X:
    Y = W[3] + Y
print(Y / 200)

print(len(X))
print('answer2', len(X[0]))


# функция, принимающая на вход Х и w, возвращая Y
def linear_prediction(X, W):
    y_res = np.linspace(0, 0, 200)
    for i in range(len(X)):
        for j in range(0, len(W) - 1):
            y_res[i] = y_res[i] + X[i, j] * W[j]
        y_res[i] = y_res[i] + W[3]
    return y_res


# сркв ошибка прогноза значений Sales с весами из нормального уравнения (W)
y_result = linear_prediction(X, W)
# print('yr=', y_result)
print('answer3', mercer(y, y_result))
print(stochastic_gradient_step(X, y, normal_equation(X, y), 0, eta=0.01))
print(X.shape[0])
print(len(X[0]))
weigths, errors = stochastic_gradient_descent(X=X, y=y, w_init=(np.zeros(4)), max_iter=100000)
print('итоговые веса', weigths)

#сркв ошибка на последней итерации
print('answer4=',errors[-1])
plot.plot((range(len(errors)), errors))
xlabel('Iteration number')
ylabel('MSE')

#plot.show()
