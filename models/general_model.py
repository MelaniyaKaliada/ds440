import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle


data = pd.read_csv("C:\\Users\\00000\\PycharmProjects\\ds440\\data\\alzheimers_prediction_dataset.csv")

X = data.drop("Alzheimer", axis=1)
y = data["Alzheimer"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=50)

# LabelEncoder
le_gender = LabelEncoder()
X_train["Gender"] = le_gender.fit_transform(X_train["Gender"])
X_test["Gender"] = le_gender.transform(X_test["Gender"])

le_diabetes = LabelEncoder()
X_train["Diabetes"] = le_diabetes.fit_transform(X_train["Diabetes"])
X_test["Diabetes"] = le_diabetes.transform(X_test["Diabetes"])

le_cholesterol = LabelEncoder()
X_train["Cholesterol Level"] = le_cholesterol.fit_transform(X_train["Cholesterol Level"])
X_test["Cholesterol Level"] = le_cholesterol.transform(X_test["Cholesterol Level"])

le_hypertension = LabelEncoder()
X_train["Hypertension"] = le_hypertension.fit_transform(X_train["Hypertension"])
X_test["Hypertension"] = le_hypertension.transform(X_test["Hypertension"])

le_family = LabelEncoder()
X_train["Family History"] = le_family.fit_transform(X_train["Family History"])
X_test["Family History"] = le_family.transform(X_test["Family History"])

le_genetic = LabelEncoder()
X_train["Genetic Risk Factor"] = le_genetic.fit_transform(X_train["Genetic Risk Factor"])
X_test["Genetic Risk Factor"] = le_genetic.transform(X_test["Genetic Risk Factor"])

le_rural_urban = LabelEncoder()
X_train["Urban vs Rural Living"] = le_rural_urban.fit_transform(X_train["Urban vs Rural Living"])
X_test["Urban vs Rural Living"] = le_rural_urban.transform(X_test["Urban vs Rural Living"])

le_alzheimer = LabelEncoder()
y_train = le_alzheimer.fit_transform(y_train)
y_test = le_alzheimer.transform(y_test)

level_order = [["Low", "Medium", "High"]]
air_enc = OrdinalEncoder(categories=level_order)
social_enc = OrdinalEncoder(categories=level_order)
stress_enc = OrdinalEncoder(categories=level_order)

X_train["Air Pollution Exposure"] = air_enc.fit_transform(X_train[["Air Pollution Exposure"]])
X_test["Air Pollution Exposure"] = air_enc.transform(X_test[["Air Pollution Exposure"]])

X_train["Social Engagement Level"] = social_enc.fit_transform(X_train[["Social Engagement Level"]])
X_test["Social Engagement Level"] = social_enc.transform(X_test[["Social Engagement Level"]])

X_train["Stress Levels"] = stress_enc.fit_transform(X_train[["Stress Levels"]])
X_test["Stress Levels"] = stress_enc.transform(X_test[["Stress Levels"]])

sleep_order = [["Poor", "Average", "Good"]]
sleep_enc = OrdinalEncoder(categories=sleep_order)

X_train["Sleep Quality"] = sleep_enc.fit_transform(X_train[["Sleep Quality"]])
X_test["Sleep Quality"] = sleep_enc.transform(X_test[["Sleep Quality"]])



X_train_index = X_train.index
X_test_index = X_test.index

# OneHotEncoder
enc_smoking = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
enc_alcohol = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
enc_diet = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
enc_marital_status = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
enc_employment = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")

smoking_encoded = enc_smoking.fit_transform(X_train[["Smoking Status"]])
smoking_encoded_test = enc_smoking.transform(X_test[["Smoking Status"]])

alcohol_encoded = enc_alcohol.fit_transform(X_train[["Alcohol Consumption"]])
alcohol_encoded_test = enc_alcohol.transform(X_test[["Alcohol Consumption"]])

diet_encoded = enc_diet.fit_transform(X_train[["Dietary Habits"]])
diet_encoded_test = enc_diet.transform(X_test[["Dietary Habits"]])

marital_encoded = enc_marital_status.fit_transform(X_train[["Marital Status"]])
marital_encoded_test = enc_marital_status.transform(X_test[["Marital Status"]])

employment_encoded = enc_employment.fit_transform(X_train[["Employment Status"]])
employment_encoded_test = enc_employment.transform(X_test[["Employment Status"]])

# Create dataframes
smoking_df = pd.DataFrame(smoking_encoded, columns=enc_smoking.get_feature_names_out(["Smoking Status"]), index=X_train_index)
smoking_df_test = pd.DataFrame(smoking_encoded_test, columns=enc_smoking.get_feature_names_out(["Smoking Status"]), index=X_test_index)

alcohol_df = pd.DataFrame(alcohol_encoded, columns=enc_alcohol.get_feature_names_out(["Alcohol Consumption"]), index=X_train_index)
alcohol_df_test = pd.DataFrame(alcohol_encoded_test, columns=enc_alcohol.get_feature_names_out(["Alcohol Consumption"]), index=X_test_index)

diet_df = pd.DataFrame(diet_encoded, columns=enc_diet.get_feature_names_out(["Dietary Habits"]), index=X_train_index)
diet_df_test = pd.DataFrame(diet_encoded_test, columns=enc_diet.get_feature_names_out(["Dietary Habits"]), index=X_test_index)

marital_df = pd.DataFrame(marital_encoded, columns=enc_marital_status.get_feature_names_out(["Marital Status"]), index=X_train_index)
marital_df_test = pd.DataFrame(marital_encoded_test, columns=enc_marital_status.get_feature_names_out(["Marital Status"]), index=X_test_index)

employment_df = pd.DataFrame(employment_encoded, columns=enc_employment.get_feature_names_out(["Employment Status"]), index=X_train_index)
employment_df_test = pd.DataFrame(employment_encoded_test, columns=enc_employment.get_feature_names_out(["Employment Status"]), index=X_test_index)

# Merge with original dataset
X_train = pd.concat([X_train, smoking_df, alcohol_df, diet_df, marital_df, employment_df], axis=1)
X_test = pd.concat([X_test, smoking_df_test, alcohol_df_test, diet_df_test, marital_df_test, employment_df_test], axis=1)


columns_to_drop = ["Smoking Status", "Alcohol Consumption", "Dietary Habits", "Marital Status", "Employment Status"]


X_train = X_train.drop(columns=columns_to_drop)
X_test = X_test.drop(columns=columns_to_drop)

X_train = X_train.drop(["Country"], axis=1)
X_test = X_test.drop(["Country"], axis=1)

columns_to_keep = ["Age", "Family History", "Employment Status_Retired", "Diabetes", "Smoking Status_Former",
    "Smoking Status_Never", "Stress Levels", "Social Engagement Level", "Air Pollution Exposure","Alcohol Consumption_Occasionally",
    "Dietary Habits_Unhealthy", "Sleep Quality", "Education Level", "Urban vs Rural Living"]


X_train = X_train[columns_to_keep]
X_test = X_test[columns_to_keep]

rf_model = RandomForestClassifier(random_state=50, criterion="entropy", max_depth=10, max_features="sqrt", min_samples_leaf=4, min_samples_split=10, n_estimators=300)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = rf_model.score(X_test, y_test)
print(f"Random Forest Accuracy:", rf_accuracy)
print(classification_report(y_test, rf_pred))


gen_model_data = {
    "model": rf_model,
    "encoders": {
        "Diabetes": le_diabetes,
        "Family History": le_family,
        "Urban vs Rural Living": le_rural_urban,
        "Air Pollution Exposure": air_enc,
        "Social Engagement Level": social_enc,
        "Stress Levels": stress_enc,
        "Sleep Quality": sleep_enc,
        "Smoking Status": enc_smoking,
        "Alcohol Consumption": enc_alcohol,
        "Dietary Habits": enc_diet,
        "Employment Status": enc_employment
    }}

with open("models/general_model.pkl", "wb") as f:
    pickle.dump(gen_model_data, f)

