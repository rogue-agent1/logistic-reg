#!/usr/bin/env python3
"""logistic_reg: Logistic regression classifier."""
import math, sys

def sigmoid(z):
    if z >= 0: return 1 / (1 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1 + ez)

class LogisticRegression:
    def __init__(self, n_features):
        self.weights = [0.0] * n_features
        self.bias = 0.0

    def predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return sigmoid(z)

    def predict(self, x):
        return 1 if self.predict_proba(x) >= 0.5 else 0

    def fit(self, X, y, lr=0.1, epochs=1000):
        n = len(X)
        for _ in range(epochs):
            for j in range(len(self.weights)):
                grad = sum((self.predict_proba(X[i]) - y[i]) * X[i][j] for i in range(n)) / n
                self.weights[j] -= lr * grad
            grad_b = sum(self.predict_proba(X[i]) - y[i] for i in range(n)) / n
            self.bias -= lr * grad_b

    def accuracy(self, X, y):
        preds = [self.predict(x) for x in X]
        return sum(1 for p, yi in zip(preds, y) if p == yi) / len(y)

    def log_loss(self, X, y):
        eps = 1e-15
        total = 0
        for xi, yi in zip(X, y):
            p = max(min(self.predict_proba(xi), 1-eps), eps)
            total += yi * math.log(p) + (1-yi) * math.log(1-p)
        return -total / len(y)

def test():
    X = [[0,0],[0,1],[1,0],[1,1],[5,5],[5,6],[6,5],[6,6]]
    y = [0,0,0,0,1,1,1,1]
    model = LogisticRegression(2)
    model.fit(X, y, lr=0.5, epochs=500)
    assert model.predict([0.5, 0.5]) == 0
    assert model.predict([5.5, 5.5]) == 1
    assert model.accuracy(X, y) == 1.0
    # Probabilities
    p0 = model.predict_proba([0, 0])
    p1 = model.predict_proba([6, 6])
    assert p0 < 0.5
    assert p1 > 0.5
    # Sigmoid
    assert abs(sigmoid(0) - 0.5) < 0.001
    assert sigmoid(10) > 0.99
    assert sigmoid(-10) < 0.01
    print("All tests passed!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test": test()
    else: print("Usage: logistic_reg.py test")
