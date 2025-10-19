import numpy as np
import matplotlib.pyplot as plt

X = np.array([0,1.25,2.5,3,4.3,5.4,8.7,10.4,12.58,13.75,14.2,19.2,21.5])
y_true = np.array([1,3.8,5.4,7.2,8.3,6.2,9.4,12.2,14.67,15.28,18.7,20.3,23.4])

X = X.reshape(-1, 1)
y_true = y_true.reshape(-1, 1)
n_samples = len(X)

def forward(x, w, b):
    return w * x + b

def loss(y_true, y_hat):
    return np.mean((y_true - y_hat) ** 2)

def gradients(X, y_true, y_hat, w, b):
    n = len(X)
    dw = (-2 / n) * np.sum(X * (y_true - y_hat))
    db = (-2 / n) * np.sum(y_true - y_hat)
    return dw, db
settings=[
    {"lr":0.5,"epochs":1000},
    {"lr":0.01,"epochs":1000},
    {"lr":0.001,"epochs":1000},
]

def train_and_plot(lr, epochs):
    w, b = 0, 0
    history = []

    print(f"\n--- 学习率: {lr}, 总训练轮数: {epochs} ---")
    for epoch in range(epochs):
        y_hat = forward(X, w, b)
        current_loss = loss(y_true, y_hat)
        history.append((w, b, current_loss))
        dw, db = gradients(X, y_true, y_hat, w, b)
        w = w - lr * dw
        b = b - lr * db

        if epoch in [0, 1, 2, 49, 99, 499, 999]:
            print(f"Epoch {epoch + 1}: w={w:.4f}, b={b:.4f}, Loss={current_loss:.4f}")

    plt.figure()
    plt.scatter(X, y_true, label='real data')
    plt.plot(X, forward(X, w, b), color='red', label=f'fit line (w={w:.4f}, b={b:.4f})')
    plt.title(f'learning rate={lr} s linear regression fitting')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
for cfg in settings:
    train_and_plot(cfg["lr"],cfg["epochs"])
