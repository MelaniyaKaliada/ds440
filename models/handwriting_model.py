import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


data = pd.read_csv("DARWIN.csv")

X = data.drop(columns=["class", "ID"])
y = data["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)



pca = PCA(n_components=88)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# rf = RandomForestClassifier(random_state=50, max_depth=9)
xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=50, learning_rate=0.05, max_depth=2, subsample=0.5)
xgb.fit(X_train, y_train)


# # Train and evaluate Random Forest
# rf.fit(X_train, y_train)
# y_pred_rf = rf.predict(X_test)
# rf_accuracy = rf.score(X_test, y_test)
# print(f"Random forest Accuracy: {rf_accuracy:.4f}")
# print("Random Forest Performance:")
# print(classification_report(y_test, y_pred_rf))

#Train and evaluate XGBoost
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
xgb_accuracy = xgb.score(X_test, y_test)
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
print("XGBoost Performance:")
print(classification_report(y_test, y_pred_xgb))

