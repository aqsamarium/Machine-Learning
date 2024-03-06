import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import KernelPCA

# --- SECTION 1 ---
# Libraries and data loading
from sklearn.linear_model import  LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
bc = load_breast_cancer()

# --- SECTION 2 ---
# Split the data into train and test set
train_x, train_y = bc.data[:400], bc.target[:400]
test_x, test_y = bc.data[400:], bc.target[400:]

# --- SECTION 3 ---
# Instantiate, train and evaluate the model
logit = LogisticRegression()
logit.fit(train_x, train_y)
acc = metrics.accuracy_score(test_y, logit.predict(test_x))

# --- SECTION 4 ---
# Print the model
print('---Logistic Regression on breast cancer dataset.---')
print('Coefficients:')
print('Intercept (b): %.2f'%logit.intercept_)
for i in range(len(bc.feature_names)):
    print(bc.feature_names[i]+': %.2f'%logit.coef_[0][i])
print('-'*30)
print('Accuracy: %.2f \n'%acc)
print(metrics.confusion_matrix(test_y, logit.predict(test_x)))