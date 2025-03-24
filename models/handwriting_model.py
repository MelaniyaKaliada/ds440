import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


data = pd.read_csv("C:\\Users\\00000\\PycharmProjects\\ds440\\data\\DARWIN.csv")

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


rf = RandomForestClassifier(random_state=50, max_depth=9)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_accuracy = rf.score(X_test, y_test)
print("Random forest Accuracy:", rf_accuracy)
print(classification_report(y_test, y_pred_rf))



