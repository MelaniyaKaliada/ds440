from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)

with open("models/general_model.pkl", "rb") as f:
    general_model_data = pickle.load(f)

general_model = general_model_data["model"]
encoders = general_model_data["encoders"]

@app.route('/')
def home():
    return render_template("Homepage.html")

@app.route('/mri')
def mri_page():
    return render_template("MRI_page.html")
@app.route('/general', methods=['GET', 'POST'])
def general_page():
    if request.method == 'GET':
        return (render_template("general_page.html"))

    if request.method == 'POST':
        form_data = {
            "Age": int(request.form["age"]),
            "Family History": request.form["family_history"],
            "Employment Status": request.form["employment_status"],
            "Diabetes": request.form["diabetes"],
            "Smoking Status": request.form["smoking_status"],
            "Stress Levels": request.form["stress_level"],
            "Social Engagement Level": request.form["social_engagement_level"],
            "Air Pollution Exposure": request.form["air_pollution_exposure"],
            "Alcohol Consumption": request.form["alcohol_consumption"],
            "Dietary Habits": request.form["dietary_habits"],
            "Sleep Quality": request.form["sleep_quality"],
            "Education Level": int(request.form["education_level"]),
            "Urban vs Rural Living": request.form["urban_rural"]
        }

        input_df = pd.DataFrame([form_data])

        # Apply Label Encoding
        label_cols = ["Diabetes", "Family History", "Urban vs Rural Living"]
        for col in label_cols:
            input_df[col] = encoders[col].transform(input_df[[col]])

        ordinal_cols = ["Air Pollution Exposure", "Social Engagement Level", "Stress Levels"]
        for col in ordinal_cols:
            input_df[col] = encoders[col].transform(input_df[[col]])

        input_df["Sleep Quality"] = encoders["Sleep Quality"].transform(input_df[["Sleep Quality"]])

        onehotenc_cols = ["Smoking Status", "Alcohol Consumption", "Dietary Habits", "Employment Status"]
        for col in onehotenc_cols:
            enc_cols = encoders[col].transform(input_df[[col]])
            enc_df = pd.DataFrame(enc_cols, columns=encoders[col].get_feature_names_out([col]))
            enc_df.index = input_df.index
            input_df = input_df.drop(col, axis=1)
            input_df = pd.concat([input_df, enc_df], axis=1)

        columns_to_keep = ["Age", "Family History", "Employment Status_Retired", "Diabetes", "Smoking Status_Former",
    "Smoking Status_Never", "Stress Levels", "Social Engagement Level", "Air Pollution Exposure","Alcohol Consumption_Occasionally",
    "Dietary Habits_Unhealthy", "Sleep Quality", "Education Level", "Urban vs Rural Living"]
        input_df = input_df[columns_to_keep]

        pred = general_model.predict(input_df)[0]

        return render_template("general_page.html", prediction=pred)

@app.route('/handwriting')
def handwriting_page():
    return render_template("Handwriting_page.html")


if __name__ == "__main__":
    app.run(debug=True)

