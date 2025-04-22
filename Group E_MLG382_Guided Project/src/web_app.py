# python src/web_app.py
import os
import pandas as pd
import numpy as np
import joblib
from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(BASE_DIR, "..", "artifacts")

# === Load Artifacts ===
scaler = joblib.load(os.path.join(ARTIFACTS, "scaler.pkl"))
feature_names = joblib.load(os.path.join(ARTIFACTS, "feature_names.pkl"))
label_encoder = joblib.load(os.path.join(ARTIFACTS, "label_encoder.pkl"))
model_rf = joblib.load(os.path.join(ARTIFACTS, "model_rf.pkl"))
performance_df = pd.read_csv(os.path.join(ARTIFACTS, "model_performance.csv"))

# === Label Mapping ===
grade_label_map = {0: "Fail", 1: "Pass", 2: "Good", 3: "Excellent"}

# === UI Options ===
categorical_options = {
    "Gender": ["Male", "Female"],
    "Ethnicity": ["Asian", "Black", "Hispanic", "White", "Other"],
    "ParentalEducation": ["High School", "Bachelor", "Master", "PhD", "None"],
    "Tutoring": ["Yes", "No"],
    "ParentalSupport": ["Yes", "No"],
    "Extracurricular": ["Yes", "No"],
    "Sports": ["Yes", "No"],
    "Music": ["Yes", "No"],
    "Volunteering": ["Yes", "No"]
}

# === Encoding Maps ===
gender_map = {"Male": 0.0, "Female": 1.0}
ethnicity_map = {"White": 0.0, "Hispanic": 1.0, "Black": 2.0, "Asian": 3.0, "Other": 4.0}
parent_edu_map = {"High School": 15.0, "Bachelor": 19.83, "Master": 20.5, "PhD": 21.5, "None": 10.0}

# === Preprocessing ===
def preprocess_user_input(age, gender, ethnicity, parent_edu, study_time, absences,
                          tutoring, parental_support, extracurricular, sports, music,
                          volunteering, gpa):

    gender_encoded = gender_map.get(gender, 0.0)
    ethnicity_encoded = ethnicity_map.get(ethnicity, 0.0)
    parent_edu_encoded = parent_edu_map.get(parent_edu, 15.0)
    tutoring = 1.0 if tutoring == "Yes" else 0.0
    parental_support = 1.0 if parental_support == "Yes" else 0.0
    extracurricular = 1.0 if extracurricular == "Yes" else 0.0
    sports = 1.0 if sports == "Yes" else 0.0
    music = 1.0 if music == "Yes" else 0.0
    volunteering = 1.0 if volunteering == "Yes" else 0.0

    numeric_scaled = scaler.transform([[study_time, absences, gpa]])

    return np.array([[
        age, gender_encoded, ethnicity_encoded, parent_edu_encoded,
        numeric_scaled[0][0],  # study time
        numeric_scaled[0][1],  # absences
        tutoring, parental_support, extracurricular,
        sports, music, volunteering,
        numeric_scaled[0][2]   # GPA
    ]])

# === Input Row Builder ===
def create_input_row(label, id, input_type="text", options=None, default=None):
    if options:
        return dbc.Row([
            dbc.Col(html.Label(label, className="fw-bold"), width=3),
            dbc.Col(dcc.Dropdown(
                id=id,
                options=[{"label": opt, "value": opt} for opt in options],
                value=default,
                placeholder=f"Select {label}"
            ), width=9)
        ], className="mb-2")
    else:
        return dbc.Row([
            dbc.Col(html.Label(label, className="fw-bold"), width=3),
            dbc.Col(dcc.Input(id=id, type=input_type, value=default, placeholder=f"Enter {label}", className="form-control"), width=9)
        ], className="mb-2")

# === Dash App Init ===
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "BrightPath Grade Prediction Dashboard"

# === Layout ===
app.layout = dbc.Container([
    html.H2("📊 Model Performance Dashboard", className="text-center my-4 fw-bold"),

    dcc.Tabs([
        dcc.Tab(label="Model Performance", children=[
            dcc.Graph(figure=px.bar(performance_df, x="Model", y="Accuracy", text="Accuracy", color="Model")
                      .update_layout(title="Model Accuracy", yaxis_range=[0, 1])),
            dcc.Graph(figure=px.bar(performance_df, x="Model", y="Precision", color="Model", title="Precision")),
            dcc.Graph(figure=px.bar(performance_df, x="Model", y="Recall", color="Model", title="Recall")),
            dcc.Graph(figure=px.bar(performance_df, x="Model", y="F1", color="Model", title="F1 Score")),
        ]),

        dcc.Tab(label="Predictions", children=[
            html.H4("🎓 Predict Learner's Grade", className="my-4 fw-bold"),
            dbc.Form([
                create_input_row("Age", "age", "number", default=18),
                create_input_row("Gender", "gender", options=categorical_options["Gender"], default="Female"),
                create_input_row("Ethnicity", "ethnicity", options=categorical_options["Ethnicity"], default="White"),
                create_input_row("ParentalEducation", "parentaleducation", options=categorical_options["ParentalEducation"], default="Bachelor"),
                create_input_row("Tutoring", "tutoring", options=categorical_options["Tutoring"], default="Yes"),
                create_input_row("ParentalSupport", "parentalsupport", options=categorical_options["ParentalSupport"], default="Yes"),
                create_input_row("Extracurricular", "extracurricular", options=categorical_options["Extracurricular"], default="Yes"),
                create_input_row("Sports", "sports", options=categorical_options["Sports"], default="No"),
                create_input_row("Music", "music", options=categorical_options["Music"], default="No"),
                create_input_row("Volunteering", "volunteering", options=categorical_options["Volunteering"], default="No"),
                create_input_row("StudyTimeWeekly", "studytimeweekly", "number", default=10),
                create_input_row("Absences", "absences", "number", default=2),
                create_input_row("GPA", "gpa", "number", default=3.2),
                dbc.Button("Predict Grade", id="predict-btn", color="primary", className="mt-3 w-100")
            ]),
            html.Div(id="prediction-output", className="mt-4 fs-5 fw-bold text-success")
        ])
    ])
], fluid=True)

# === Callback ===
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    [
        State("age", "value"), State("gender", "value"), State("ethnicity", "value"),
        State("parentaleducation", "value"), State("studytimeweekly", "value"),
        State("absences", "value"), State("tutoring", "value"), State("parentalsupport", "value"),
        State("extracurricular", "value"), State("sports", "value"),
        State("music", "value"), State("volunteering", "value"), State("gpa", "value")
    ]
)
def predict_rf_only(n_clicks, age, gender, ethnicity, parental_education, study_time, absences,
                    tutoring, parental_support, extracurricular, sports, music, volunteering, gpa):
    if not n_clicks:
        return ""
    
    X_array = preprocess_user_input(
        age, gender, ethnicity, parental_education, study_time,
        absences, tutoring, parental_support, extracurricular,
        sports, music, volunteering, gpa
    )

    X_input = pd.DataFrame(X_array, columns=feature_names)
    
    pred = model_rf.predict(X_input)[0]
    label = grade_label_map.get(int(pred), "Unknown")

    return html.Div([
        html.H5("🎯 Predicted Grade using Random Forest:"),
        html.P(f"{label}", className="text-success fs-4 mt-2")
    ])

# === Run Server ===
if __name__ == "__main__":
    app.run(debug=True)
