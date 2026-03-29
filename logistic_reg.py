#!/usr/bin/env python3
"""logistic_reg - Logistic regression."""
import sys,argparse,json,math,random
def sigmoid(x):return 1/(1+math.exp(-max(-500,min(500,x))))
class LogisticRegression:
    def __init__(self,n_features,lr=0.1):
        self.weights=[0]*n_features;self.bias=0;self.lr=lr
    def predict_proba(self,x):return sigmoid(sum(w*xi for w,xi in zip(self.weights,x))+self.bias)
    def predict(self,x):return 1 if self.predict_proba(x)>0.5 else 0
    def train(self,X,y,epochs=100):
        losses=[]
        for ep in range(epochs):
            total_loss=0
            for xi,yi in zip(X,y):
                p=self.predict_proba(xi);error=p-yi
                for j in range(len(self.weights)):self.weights[j]-=self.lr*error*xi[j]
                self.bias-=self.lr*error
                total_loss+=-(yi*math.log(max(p,1e-10))+(1-yi)*math.log(max(1-p,1e-10)))
            losses.append(round(total_loss/len(X),6))
        return losses
def main():
    p=argparse.ArgumentParser(description="Logistic regression")
    p.add_argument("--samples",type=int,default=200);p.add_argument("--epochs",type=int,default=100)
    args=p.parse_args()
    random.seed(42)
    X=[];y=[]
    for _ in range(args.samples//2):
        X.append([random.gauss(2,1),random.gauss(2,1)]);y.append(0)
        X.append([random.gauss(5,1),random.gauss(5,1)]);y.append(1)
    split=int(len(X)*0.8)
    lr=LogisticRegression(2);losses=lr.train(X[:split],y[:split],args.epochs)
    correct=sum(1 for xi,yi in zip(X[split:],y[split:]) if lr.predict(xi)==yi)
    acc=correct/len(X[split:])
    print(json.dumps({"epochs":args.epochs,"final_loss":losses[-1],"accuracy":round(acc,4),"weights":[round(w,4) for w in lr.weights],"bias":round(lr.bias,4)},indent=2))
if __name__=="__main__":main()
