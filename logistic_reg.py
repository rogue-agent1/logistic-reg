#!/usr/bin/env python3
"""Logistic regression from scratch with gradient descent."""
import sys, math, random

def sigmoid(x): return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))

class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr; self.n_iters = n_iters; self.w = None; self.b = 0

    def fit(self, X, y):
        n, d = len(X), len(X[0])
        self.w = [0.0] * d
        for epoch in range(self.n_iters):
            dw = [0.0] * d; db = 0
            for i in range(n):
                z = sum(self.w[j]*X[i][j] for j in range(d)) + self.b
                pred = sigmoid(z); err = pred - y[i]
                for j in range(d): dw[j] += err * X[i][j]
                db += err
            for j in range(d): self.w[j] -= self.lr * dw[j] / n
            self.b -= self.lr * db / n

    def predict_proba(self, x):
        z = sum(self.w[j]*x[j] for j in range(len(x))) + self.b
        return sigmoid(z)

    def predict(self, x): return 1 if self.predict_proba(x) >= 0.5 else 0

    def score(self, X, y):
        return sum(1 for xi, yi in zip(X, y) if self.predict(xi) == yi) / len(y)

    def log_loss(self, X, y):
        eps = 1e-15; n = len(y)
        return -sum(yi*math.log(max(self.predict_proba(xi), eps)) + (1-yi)*math.log(max(1-self.predict_proba(xi), eps)) for xi, yi in zip(X, y)) / n

def main():
    random.seed(42)
    X = [[random.gauss(-1.5, 1), random.gauss(-1.5, 1)] for _ in range(50)] +         [[random.gauss(1.5, 1), random.gauss(1.5, 1)] for _ in range(50)]
    y = [0]*50 + [1]*50
    idx = list(range(100)); random.shuffle(idx)
    X = [X[i] for i in idx]; y = [y[i] for i in idx]
    lr = LogisticRegression(lr=0.1, n_iters=500); lr.fit(X[:80], y[:80])
    print(f"Logistic Regression")
    print(f"Weights: [{', '.join(f'{w:.4f}' for w in lr.w)}], bias: {lr.b:.4f}")
    print(f"Train accuracy: {lr.score(X[:80], y[:80])*100:.1f}%")
    print(f"Test accuracy: {lr.score(X[80:], y[80:])*100:.1f}%")
    print(f"Log loss: {lr.log_loss(X[80:], y[80:]):.4f}")

if __name__ == "__main__": main()
