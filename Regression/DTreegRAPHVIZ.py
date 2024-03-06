import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import KernelPCA

# --- SECTION 1 ---
# Libraries and data loading
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
bc = load_breast_cancer()



# --- SECTION 2 ---
# Split the data into train and test set
train_x, train_y = bc.data[:400], bc.target[:400]
test_x, test_y = bc.data[400:], bc.target[400:]

# --- SECTION 3 ---
# Instantiate, train and evaluate the model
dtc = DecisionTreeClassifier(max_depth=2)
dtc.fit(train_x, train_y)
acc = metrics.accuracy_score(test_y, dtc.predict(test_x))

# --- SECTION 4 ---
# Print the model's accuracy
print('---Neural Networks on breast cancer dataset.---')
print('Accuracy: %.2f \n'%acc)
print(metrics.confusion_matrix(test_y, dtc.predict(test_x)))
from sklearn.tree import export_graphviz
export_graphviz(dtc, feature_names=bc.feature_names,
                             class_names=bc.target_names, impurity=False)