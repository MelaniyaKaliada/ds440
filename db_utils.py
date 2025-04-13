import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'predictions.db')

def log_general_prediction(data, prediction):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute('''
        INSERT INTO general_predictions (
            age, family_history, employment_status, diabetes, smoking_status,
            stress_levels, social_engagement_level, air_pollution_exposure,
            alcohol_consumption, dietary_habits, sleep_quality, education_level,
            urban_rural, prediction
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data["Age"], data["Family History"], data["Employment Status"], data["Diabetes"],
        data["Smoking Status"], data["Stress Levels"], data["Social Engagement Level"],
        data["Air Pollution Exposure"], data["Alcohol Consumption"], data["Dietary Habits"],
        data["Sleep Quality"], data["Education Level"], data["Urban vs Rural Living"], prediction
    ))

    conn.commit()
    conn.close()

def log_mri_prediction(data, prediction):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute('''
        INSERT INTO mri_predictions (
            gender, age, education, ses, mmse, etiv, nwbv, asf, cdr, prediction
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data["M/F"], data["Age"], data["EDUC"], data["SES"],
        data["MMSE"], data["eTIV"], data["nWBV"], data["ASF"], data["CDR"], prediction
    ))

    conn.commit()
    conn.close()