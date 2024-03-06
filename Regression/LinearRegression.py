
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import KernelPCA

# --- SECTION 1 ---
# Libraries and data loading
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn import metrics
diabetes = load_diabetes()


# --- SECTION 2 ---
# Split the data into train and test set
train_x, train_y = diabetes.data[:400], diabetes.target[:400]
test_x, test_y = diabetes.data[400:], diabetes.target[400:]

# --- SECTION 3 ---
# Instantiate, train and evaluate the model
ols = LinearRegression()
ols.fit(train_x, train_y)
err = metrics.mean_squared_error(test_y, ols.predict(test_x))
r2 = metrics.r2_score(test_y, ols.predict(test_x))

# --- SECTION 4 ---
# Print the model
print('---OLS on diabetes dataset.---')
print('Coefficients:')
print('Intercept (b): %.2f'%ols.intercept_)
for i in range(len(diabetes.feature_names)):
    print(diabetes.feature_names[i]+': %.2f'%ols.coef_[i])
print('-'*30)
print('R-squared: %.2f'%r2, ' MSE: %.2f \n'%err)
