import numpy as np

df = np.array([
    [12.0, 1.5, 1, 'Wine'],
    [5.0, 2.0, 0, 'Beer'],
    [40.0, 0.0, 1, 'Whiskey'],
    [13.5, 1.2, 1, 'Wine'],
    [4.5, 1.8, 0, 'Beer'],
    [38.0, 0.1, 1, 'Whiskey'],
    [11.5, 1.7, 1, 'Wine'],
    [5.5, 2.3, 0, 'Beer']
])

df2 = np.array([
    [6.0, 2.1, 0],   # Expected: Beer
    [39.0, 0.05, 1], # Expected: Whiskey
    [13.0, 1.3, 1]   # Expected: Wine
])

labels = {'Beer': 0, 'Whiskey': 1, 'Wine': 2}

#separating the data
X = df[:, :-1].astype(float)                     
y = np.array([labels[n] for n in df[:, -1]])

#calculating gini impurity
def ginicalc(y):
    label, count = np.unique(y, return_counts=True)
    probs = count/len(y)
    return 1 - np.sum(probs**2)

#finding the best split
def bestsplit(X, y):
    n_samples, n_features = X.shape
    bestgini = float('inf')

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature]) 

        for threshold in thresholds:
            left  = X[:, feature] <= threshold
            yleft  = y[left]
            yright = y[~left]

            if len(yleft) == 0 or len(yright) == 0:
                continue

            lgini = ginicalc(yleft)
            rgini = ginicalc(yright)
            weightedgini = (len(yleft)/n_samples)*lgini + (len(yright)/n_samples)*rgini

            if weightedgini < bestgini:
                bestgini = weightedgini
                bestfeature = feature
                bestthreshold = threshold

    return bestfeature, bestthreshold, bestgini

class Node():
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
    
#building the tree
def buildtree(X, y, depth = 0, maxdepth = 4, minsamples = 2):
    n_samples = len(y)
    if(depth == maxdepth or len(np.unique(y)) == 1 or n_samples < minsamples):
        values, count = np.unique(y, return_counts=True)
        leafval = values[np.argmax(count)]
        leafnode = Node()
        leafnode.value = leafval
        return leafnode

    f, t, g = bestsplit(X, y)
    left  = X[:, f] <= t
    right = ~left

    leftchild  = buildtree(X[left],  y[left],  depth+1, maxdepth, minsamples)
    rightchild = buildtree(X[right], y[right], depth+1, maxdepth, minsamples)

    node = Node()
    node.feature_index = f
    node.threshold = t
    node.left = leftchild
    node.right = rightchild

    return node

#traversing each node
def oneprediction(node, x):
    if(node.value != None):
        return node.value
    
    if(x[node.feature_index] <= node.threshold):
        return oneprediction(node.left, x)
    else:
        return oneprediction(node.right, x)

#final prediction procedure
def predict(node, X):
    finalvals = np.array([oneprediction(node, x) for x in X])
    return finalvals

print(predict(buildtree(X, y), df2))

