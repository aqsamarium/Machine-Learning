import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import KernelPCA 

# --- SECTION 1 ---
# Libraries and data loading
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
bc = load_breast_cancer()

# --- SECTION 2 ---
# Split the data into train and test set
train_x, train_y = bc.data[:400], bc.target[:400]
test_x, test_y = bc.data[400:], bc.target[400:]

# --- SECTION 3 ---
# Instantiate, train and evaluate the model
svc = SVC(kernel='linear')
svc.fit(train_x, train_y)
acc = metrics.accuracy_score(test_y, svc.predict(test_x))

# --- SECTION 4 ---
# Print the model's accuracy
print('---SVM on breast cancer dataset.---')
print('Accuracy: %.2f \n'%acc)
print(metrics.confusion_matrix(test_y, svc.predict(test_x)))
