import sqlite3

conn = sqlite3.connect('predictions.db')
cur = conn.cursor()

cur.execute('''
CREATE TABLE IF NOT EXISTS general_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    family_history TEXT,
    employment_status TEXT,
    diabetes TEXT,
    smoking_status TEXT,
    stress_levels TEXT,
    social_engagement_level TEXT,
    air_pollution_exposure TEXT,
    alcohol_consumption TEXT,
    dietary_habits TEXT,
    sleep_quality TEXT,
    education_level INTEGER,
    urban_rural TEXT,
    prediction TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')


cur.execute('''
CREATE TABLE IF NOT EXISTS mri_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gender TEXT,
    age INTEGER,
    education INTEGER,
    ses TEXT,
    mmse INTEGER,
    etiv INTEGER,
    nwbv REAL,
    asf REAL,
    cdr REAL,
    prediction TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

conn.commit()
conn.close()