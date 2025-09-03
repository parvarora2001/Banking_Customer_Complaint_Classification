import joblib

MODEL_PATH = "artifacts/model.joblib"
text = "I applied for a mortgage but the bank kept delaying my application and then denied it without a clear explanation. They also pulled my credit report multiple times without consent, which hurt my credit score."

class ComplaintModel:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

    def predict(self, text: str):
        return self.model.predict([text])[0]

# Singleton
model = ComplaintModel()
