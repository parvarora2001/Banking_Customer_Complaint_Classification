from fastapi import FastAPI
from pydantic import BaseModel
from app.model import model

app = FastAPI(title="Complaint Classification API")

class ComplaintRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: ComplaintRequest):
    prediction = model.predict(request.text)
    return {"prediction": prediction}
