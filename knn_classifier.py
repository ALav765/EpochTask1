import numpy as np
import matplotlib as plt

df = np.array([
    [150, 7.0, 1, 'Apple'],
    [120, 6.5, 0, 'Banana'],
    [180, 7.5, 2, 'Orange'],
    [155, 7.2, 1, 'Apple'],
    [110, 6.0, 0, 'Banana'],
    [190, 7.8, 2, 'Orange'],
    [145, 7.1, 1, 'Apple'],
    [115, 6.3, 0, 'Banana']
])

labels = {'Apple': 0, 'Banana': 1, 'Orange': 2}
X = df[:, :-1].astype(float)                     
y = np.array([labels[n] for n in df[:, -1]])

X_normalized = (X - X.mean(axis = 0))/X.std(axis = 0)

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a-b)**2))

class knn():
    def __init__(self, k):
        self.k = k
    def fit(self, X_normalized, y):
        self.X_normalized = X_normalized
        self.y = y
    def predict_one(self, x):
        dists = [euclidean_distance(x, i) for i in X_normalized]
        nearest= np.argsort(dists)[:self.k]
        klabels = self.y[nearest]
        values, counts = np.unique(klabels, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X_test):
        return np.array([self.predict_one(x) for x in X_test])
    
    def accuracy(self, X_test, y_test):
        count = 0
        for i in range(len(y_test)):
            if np.array([self.predict_one(x) for x in Xtest_normalized])[i] == y_test[i]:
                count+=1
        return count/len(y_test)*100


knn1 = knn(k=3)
knn1.fit(X_normalized, y)


#test section
X_test = np.array([
    [118, 6.2, 0],  # Expected: Banana
    [160, 7.3, 1],  # Expected: Apple
    [185, 7.7, 2]   # Expected: Orange
])
y_test= np.array([1, 0, 2])
Xtest_normalized = (X_test - X_test.mean(axis = 0))/X_test.std(axis = 0)
print(knn1.predict(Xtest_normalized))
print(y_test)
print(f"{knn1.accuracy(X_test, y_test)}%")


