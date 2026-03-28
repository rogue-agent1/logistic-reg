#!/usr/bin/env python3
"""logistic_reg - Logistic regression from scratch."""
import sys,math,random
def sigmoid(z):return 1/(1+math.exp(-max(-500,min(500,z))))
def predict(X,w,b):return[sigmoid(sum(xi*wi for xi,wi in zip(x,w))+b) for x in X]
def train(X,y,lr=0.01,epochs=1000):
    d=len(X[0]);w=[0]*d;b=0;n=len(X)
    for _ in range(epochs):
        preds=predict(X,w,b)
        dw=[sum((preds[i]-y[i])*X[i][j] for i in range(n))/n for j in range(d)]
        db=sum(preds[i]-y[i] for i in range(n))/n
        w=[wi-lr*dwi for wi,dwi in zip(w,dw)];b-=lr*db
    return w,b
def accuracy(X,y,w,b):
    preds=predict(X,w,b);return sum(1 for p,yi in zip(preds,y) if(p>=0.5)==yi)/len(y)
if __name__=="__main__":
    random.seed(42);X=[];y=[]
    for _ in range(50):x1=random.gauss(-1,1);x2=random.gauss(-1,1);X.append([x1,x2]);y.append(0)
    for _ in range(50):x1=random.gauss(1,1);x2=random.gauss(1,1);X.append([x1,x2]);y.append(1)
    w,b=train(X,y);print(f"Weights: {[round(wi,3) for wi in w]}, Bias: {round(b,3)}")
    print(f"Accuracy: {accuracy(X,y,w,b):.1%}")
