#!/usr/bin/env python3
"""logistic_reg - Logistic regression with gradient descent."""
import sys, math

def sigmoid(z):
    if z > 500: return 1.0
    if z < -500: return 0.0
    return 1.0 / (1.0 + math.exp(-z))

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = []
        self.bias = 0

    def fit(self, X, y):
        n = len(X)
        d = len(X[0])
        self.weights = [0.0] * d
        self.bias = 0.0
        for _ in range(self.epochs):
            for i in range(n):
                z = sum(self.weights[j] * X[i][j] for j in range(d)) + self.bias
                pred = sigmoid(z)
                error = pred - y[i]
                for j in range(d):
                    self.weights[j] -= self.lr * error * X[i][j] / n
                self.bias -= self.lr * error / n

    def predict_proba(self, x):
        z = sum(self.weights[j] * x[j] for j in range(len(x))) + self.bias
        return sigmoid(z)

    def predict(self, x, threshold=0.5):
        return 1 if self.predict_proba(x) >= threshold else 0

    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(X)) if self.predict(X[i]) == y[i])
        return correct / len(X)

def test():
    X = [[0,0],[0,1],[1,0],[1,1],[2,2],[3,3],[2,3],[3,2]]
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    lr = LogisticRegression(lr=0.5, epochs=500)
    lr.fit(X, y)
    assert lr.predict([0, 0]) == 0
    assert lr.predict([3, 3]) == 1
    acc = lr.accuracy(X, y)
    assert acc >= 0.75
    p = lr.predict_proba([0, 0])
    assert p < 0.5
    p2 = lr.predict_proba([3, 3])
    assert p2 > 0.5
    print("All tests passed!")

if __name__ == "__main__":
    test() if "--test" in sys.argv else print("logistic_reg: Logistic regression. Use --test")
