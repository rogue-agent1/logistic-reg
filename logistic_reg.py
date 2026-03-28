#!/usr/bin/env python3
"""Logistic Regression — zero-dep implementation."""
import math, random

def sigmoid(x): return 1/(1+math.exp(-max(-500,min(500,x))))

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000, reg=0.0):
        self.lr=lr; self.epochs=epochs; self.reg=reg
    def fit(self, X, y):
        n,d=len(X),len(X[0])
        self.w=[0.0]*d; self.b=0.0
        for _ in range(self.epochs):
            for i in range(n):
                p=sigmoid(sum(self.w[j]*X[i][j] for j in range(d))+self.b)
                err=y[i]-p
                for j in range(d): self.w[j]+=self.lr*(err*X[i][j]-self.reg*self.w[j])/n
                self.b+=self.lr*err/n
    def predict_proba(self, X):
        return [sigmoid(sum(self.w[j]*x[j] for j in range(len(x)))+self.b) for x in X]
    def predict(self, X, threshold=0.5):
        return [1 if p>=threshold else 0 for p in self.predict_proba(X)]

if __name__=="__main__":
    random.seed(42)
    X=[[random.gauss(-1,1),random.gauss(-1,1)] for _ in range(50)]+[[random.gauss(1,1),random.gauss(1,1)] for _ in range(50)]
    y=[0]*50+[1]*50
    lr=LogisticRegression(lr=0.5,epochs=200); lr.fit(X,y)
    preds=lr.predict(X); acc=sum(p==a for p,a in zip(preds,y))/len(y)
    print(f"Accuracy: {acc:.0%}")
    print(f"Weights: [{', '.join(f'{w:.3f}' for w in lr.w)}], Bias: {lr.b:.3f}")
