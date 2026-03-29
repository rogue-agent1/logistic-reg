#!/usr/bin/env python3
"""Logistic regression with gradient descent."""
import sys, math, random

def sigmoid(x): return 1/(1+math.exp(-max(-500,min(500,x))))

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr, self.epochs = lr, epochs
        self.w = self.b = None
    def fit(self, X, y):
        n, d = len(X), len(X[0])
        self.w = [0.0]*d
        self.b = 0.0
        for _ in range(self.epochs):
            for i in range(n):
                z = sum(self.w[j]*X[i][j] for j in range(d)) + self.b
                pred = sigmoid(z)
                err = pred - y[i]
                for j in range(d):
                    self.w[j] -= self.lr * err * X[i][j] / n
                self.b -= self.lr * err / n
    def predict_proba(self, x):
        return sigmoid(sum(self.w[j]*x[j] for j in range(len(x))) + self.b)
    def predict(self, x):
        return 1 if self.predict_proba(x) >= 0.5 else 0

def test():
    random.seed(42)
    X = [[random.gauss(2,1), random.gauss(2,1)] for _ in range(30)] +         [[random.gauss(-2,1), random.gauss(-2,1)] for _ in range(30)]
    y = [1]*30 + [0]*30
    lr = LogisticRegression(lr=0.1, epochs=500)
    lr.fit(X, y)
    correct = sum(1 for i in range(60) if lr.predict(X[i]) == y[i])
    assert correct >= 50, f"Accuracy: {correct}/60"
    p = lr.predict_proba([3, 3])
    assert p > 0.7
    p2 = lr.predict_proba([-3, -3])
    assert p2 < 0.3
    print("  logistic_reg: ALL TESTS PASSED")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test": test()
    else: print("Logistic regression")
