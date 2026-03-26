from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_workforce

app = FastAPI(title="Workforce Prediction API")

class InputData(BaseModel):
    Emp_2018: int
    Emp_2019: int
    Emp_2020: int
    Emp_2021: int
    Emp_2022: int
    Emp_2023: int
    Emp_2024: int

@app.get("/")
def home():
    return {"status": "API running"}

@app.post("/predict")
def predict(data: InputData):
    features = [
        data.Emp_2018,
        data.Emp_2019,
        data.Emp_2020,
        data.Emp_2021,
        data.Emp_2022,
        data.Emp_2023,
        data.Emp_2024
    ]
    
    prediction = predict_workforce(features)
    
    return {"predicted_2025": prediction}