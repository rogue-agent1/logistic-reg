#!/usr/bin/env python3
"""Logistic regression. Zero dependencies."""
import math

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr; self.epochs = epochs; self.weights = []; self.bias = 0

    def _sigmoid(self, z):
        return 1 / (1 + math.exp(-max(-500, min(500, z))))

    def fit(self, X, y):
        n, d = len(X), len(X[0])
        self.weights = [0.0] * d; self.bias = 0.0
        for _ in range(self.epochs):
            for i in range(n):
                z = sum(self.weights[j]*X[i][j] for j in range(d)) + self.bias
                pred = self._sigmoid(z)
                err = pred - y[i]
                for j in range(d):
                    self.weights[j] -= self.lr * err * X[i][j] / n
                self.bias -= self.lr * err / n
        return self

    def predict_proba(self, X):
        return [self._sigmoid(sum(self.weights[j]*x[j] for j in range(len(x))) + self.bias) for x in X]

    def predict(self, X, threshold=0.5):
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(1 for p, t in zip(preds, y) if p == t) / len(y)

if __name__ == "__main__":
    X = [[0,0],[1,0],[0,1],[1,1]]; y = [0,0,0,1]
    lr = LogisticRegression(lr=0.5, epochs=500).fit(X, y)
    print(f"Predictions: {lr.predict(X)}")
