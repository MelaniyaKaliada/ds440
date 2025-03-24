import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
import pickle

data = pd.read_csv("C:\\Users\\00000\\PycharmProjects\\ds440\\data\\oasis_longitudinal.csv")

median_ses = data['SES'].median()
median_mmse = data['MMSE'].median()

data['SES'] = data['SES'].fillna(median_ses)
data['MMSE'] = data['MMSE'].fillna(median_mmse)


data_selected = data[["Group", "M/F", "Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF", "CDR"]]

X = data_selected.drop('Group', axis=1)
y = data_selected['Group']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=50)

y_train = y_train.replace("Converted", "Demented")
y_test = y_test.replace("Converted", "Demented")

label_encoder_MF = LabelEncoder()
X_train['M/F'] = label_encoder_MF.fit_transform(X_train['M/F'])
X_test['M/F'] = label_encoder_MF.transform(X_test['M/F'])

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


rf_model = RandomForestClassifier(random_state=50, n_estimators = 150, max_depth=8)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = rf_model.score(X_test, y_test)
print(f"Random Forest Accuracy:", rf_accuracy)
print(classification_report(y_test, rf_pred))

mri_model_data = {
    "model": rf_model,
    "encoder": label_encoder_MF}

with open("C:\\Users\\00000\\PycharmProjects\\ds440\\models\\mri_model.pkl", "wb") as f:
    pickle.dump(mri_model_data, f)

