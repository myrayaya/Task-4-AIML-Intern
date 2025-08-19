#importing libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ( confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt

#loading in dataset
df = pd.read_csv('dataset/data.csv')
print(df.head())

df['diagnosis'] = df['diagnosis'].map({'M' : 1, 'B' : 0})

x = df.drop('diagnosis', axis = 1)
y = df['diagnosis']

# impute missing values
from sklearn.impute import SimpleImputer

x = x.loc[:, x.notna().any()]

imputer = SimpleImputer(strategy='mean')
x = pd.DataFrame(imputer.fit_transform(x), columns = x.columns)

print(x.shape)

# TRAIN TEST SPLIT
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)

# standardizing features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# fitting logistic regression model
log_reg = LogisticRegression()
log_reg.fit(x_train_scaled, y_train)

# predictions and probabilities
y_pred = log_reg.predict(x_test_scaled)
y_proba = log_reg.predict_proba(x_test_scaled)[:, 1]

# EVALUATION
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print('Confusion Matrix:')
print(cm)
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'ROC-AUC: {roc_auc:.2f}')

# ROC CURVE PLOT
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label = 'ROC Curve (area = %.2f)' % roc_auc)
plt.plot([0,1], [0,1], linestyle = '--', color = 'gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# TUNING PREDICTION THRESHOLD
custom_threshold = 0.6
y_pred_custom_thresh = (y_proba >= custom_threshold).astype(int)
cm_custom = confusion_matrix(y_test, y_pred_custom_thresh)

print(f'Confusion Matrix at threshold {custom_threshold}: ')
print(cm_custom)
print(f'Precision: {precision_score(y_test, y_pred_custom_thresh):.2f}')
print(f'Recall: {recall_score(y_test, y_pred_custom_thresh):.2f}')

# SIGMOID FUNCTION VISUALIZATION
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x_vals = np.linspace(-10, 10, 200)
sigmoid_vals = sigmoid(x_vals)

plt.figure(figsize=(7, 4))
plt.plot(x_vals, sigmoid_vals)
plt.title('Sigmoid Function')
plt.xlabel('Input (z)')
plt.ylabel('Sigmoid (z)')
plt.grid(True)
plt.show()