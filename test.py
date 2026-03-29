from logistic_reg import LogisticRegression
X = [[0,0],[0,1],[1,0],[1,1],[2,2],[3,3],[2,3],[3,2]]
y = [0,0,0,0,1,1,1,1]
lr = LogisticRegression(lr=1.0, epochs=500).fit(X, y)
preds = lr.predict(X)
assert lr.score(X, y) >= 0.75
proba = lr.predict_proba(X)
assert all(0 <= p <= 1 for p in proba)
print("logistic_reg tests passed")
