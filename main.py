import numpy as np
import pandas as pd

df = pd.read_csv("./column_2C.dat", sep=" ")
data = df.values

c1 = data[:, -1] == "AB"
c2 = data[:, -1] == "NO"

x1 = data[c1, 0:6].astype(float)
x2 = data[c2, 0:6].astype(float)


X = np.concatenate((x1, x2))

N, p = X.shape

Y = np.ones((X.shape[0], 1))
Y[c1, :] = -1

X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

rounds = 100

for i in range(rounds):
    seed = np.random.permutation(N)

    X_ = np.copy(X)[seed, :]
    Y_ = np.copy(Y)[seed, :]

    X_training = X_[0 : int(N * 0.8), :]
    Y_training = Y_[0 : int(N * 0.8), :]

    X_test = X_[int(N * 0.8) :, :]
    Y_test = Y_[int(N * 0.8) :, :]

    W = np.linalg.pinv(X_training.T @ X_training) @ X_training.T @ Y_training
    Y_predicate = X_test @ W

    desc = np.sign(Y_predicate)
    hits = np.sum(Y_test == desc)
    acc = hits / X_test.shape[0] * 100

    print(f"Accuracy: {acc:.4}%")
