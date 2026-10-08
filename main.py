from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and preprocessing files
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")


class LoanApplication(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str


@app.get("/")
def home():
    return {"message": "Loan Prediction API is running!"}


@app.post("/predict")
def predict_loan(application: LoanApplication):

    data = application.model_dump()

    df = pd.DataFrame([data])

    df = pd.get_dummies(
        df,
        columns=[
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "Property_Area"
        ],
        drop_first=True
    )

    # Match the exact columns used during model training
    df = df.reindex(columns=feature_names, fill_value=0)

    # Standardize the input
    df_scaled = scaler.transform(df)

    # Make prediction
    prediction = model.predict(df_scaled)[0]

    if prediction == 1:
        result = "Approved"
    else:
        result = "Rejected"

    return {"prediction": result}