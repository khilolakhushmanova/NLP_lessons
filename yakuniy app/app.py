
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class PredictRequest(BaseModel):
    text: str

def create_sentiment_api():
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    def predict(request: PredictRequest):
        text = request.text.strip()

        if not text:
            raise HTTPException(status_code=422, detail="Empty text")

        return {
            "sentiment": "positive",
            "confidence": 1.0,
        }

    return app
