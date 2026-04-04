import numpy as np
from collections import Counter

X_TRAIN = [[1, 2], [2, 3], [3, 3]]
Y_TRAIN = [0, 0, 1]

NEW_X = [2, 2] # predict

distances = np.sqrt(np.sum((X_TRAIN - NEW_X)**2, axis=1))

IDX = np.argsort(distances)

k = 2
nearest_idx = idx[:k]
nearest_labels = Y_TRAIN[nearest_idx]

prediction = Counter(nearest_labels).most_common(1)[0][0]

def knn_predict(X_train, y_train, x, k=3):
    distances = np.sqrt(np.sum((X_train - x)**2, axis=1))

    idx = np.argsort(distances)

    nearest_idx = idx[:k]
    nearest_labels = y_train[nearest_idx]

    return Counter(nearest_labels).most_common(1)[0][0]