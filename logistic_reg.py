#!/usr/bin/env python3
"""Logistic regression with gradient descent."""
import sys, math, random

def sigmoid(z): return 1/(1+math.exp(-max(-500,min(500,z))))

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr, self.epochs = lr, epochs
    def fit(self, X, y):
        n, d = len(X), len(X[0])
        self.w = [0.0]*d; self.b = 0.0
        for _ in range(self.epochs):
            for i in range(n):
                z = sum(self.w[j]*X[i][j] for j in range(d)) + self.b
                pred = sigmoid(z); err = pred - y[i]
                for j in range(d): self.w[j] -= self.lr * err * X[i][j]
                self.b -= self.lr * err
    def predict_proba(self, X):
        return [sigmoid(sum(self.w[j]*x[j] for j in range(len(x)))+self.b) for x in X]
    def predict(self, X): return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]
    def accuracy(self, X, y):
        preds = self.predict(X)
        return sum(p==t for p,t in zip(preds,y))/len(y)

def main():
    random.seed(42)
    X = [[random.gauss(0,1), random.gauss(0,1)] for _ in range(100)]
    y = [1 if x[0]+x[1]>0 else 0 for x in X]
    lr = LogisticRegression(lr=0.1, epochs=100); lr.fit(X, y)
    print(f"Weights: {[f'{w:.3f}' for w in lr.w]}, bias: {lr.b:.3f}")
    print(f"Train accuracy: {lr.accuracy(X, y)*100:.1f}%")

if __name__ == "__main__": main()
