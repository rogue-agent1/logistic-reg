#!/usr/bin/env python3
"""Logistic regression from scratch."""
import sys, math, random

def sigmoid(z): return 1/(1+math.exp(-max(-500,min(500,z))))

class LogisticRegression:
    def __init__(self, n_features):
        self.w = [random.gauss(0,0.01) for _ in range(n_features)]
        self.b = 0.0
    def predict_prob(self, x):
        return sigmoid(sum(wi*xi for wi,xi in zip(self.w, x)) + self.b)
    def predict(self, x):
        return 1 if self.predict_prob(x) >= 0.5 else 0
    def train(self, X, y, lr=0.1, epochs=1000):
        for epoch in range(epochs):
            loss = 0
            for xi, yi in zip(X, y):
                p = self.predict_prob(xi)
                err = p - yi
                for j in range(len(self.w)):
                    self.w[j] -= lr * err * xi[j]
                self.b -= lr * err
                loss -= yi*math.log(p+1e-10) + (1-yi)*math.log(1-p+1e-10)
            if epoch % 200 == 0:
                acc = sum(1 for xi,yi in zip(X,y) if self.predict(xi)==yi)/len(y)
                print(f"Epoch {epoch}: loss={loss/len(y):.4f} acc={acc:.1%}")

if __name__ == '__main__':
    random.seed(42)
    # Generate linearly separable data
    X, y = [], []
    for _ in range(50):
        x1, x2 = random.gauss(2,1), random.gauss(2,1); X.append([x1,x2]); y.append(1)
        x1, x2 = random.gauss(-2,1), random.gauss(-2,1); X.append([x1,x2]); y.append(0)
    lr = LogisticRegression(2)
    lr.train(X, y, lr=0.05, epochs=1000)
    acc = sum(1 for xi,yi in zip(X,y) if lr.predict(xi)==yi)/len(y)
    print(f"\nFinal accuracy: {acc:.1%}")
    print(f"Weights: {[f'{w:.3f}' for w in lr.w]}, bias: {lr.b:.3f}")
    print(f"Predict [3,3]: {lr.predict([3,3])} (prob={lr.predict_prob([3,3]):.3f})")
    print(f"Predict [-3,-3]: {lr.predict([-3,-3])} (prob={lr.predict_prob([-3,-3]):.3f})")
