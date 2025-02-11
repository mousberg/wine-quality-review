from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Optional
import joblib
import numpy as np
import os
import time
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Wine Origin Predictor API",
    description="Predicts the country of origin for wines based on their characteristics",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input validation models
class WineInput(BaseModel):
    description: str = Field(..., min_length=10, max_length=1000)
    points: float = Field(..., ge=0, le=100)
    price: float = Field(..., ge=0, le=10000)
    variety: str = Field(..., min_length=1)

    @validator('description')
    def validate_description(cls, v):
        if not v.strip():
            raise ValueError('Description cannot be empty or just whitespace')
        return v.strip()

# Response models
class PredictionResponse(BaseModel):
    predicted_country: str
    confidence_scores: Dict[str, float]
    prediction_time: float
    model_version: str = "1.0.0"
    timestamp: str

# Error response model
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str

# Update model path to be relative to the API directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Load models at startup
try:
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest_tfidf.pkl"))
    xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost_tfidf.pkl"))
    tfidf_vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    variety_encoder = joblib.load(os.path.join(MODEL_DIR, "onehot_encoder.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    
    MODELS_LOADED = True
except Exception as e:
    print(f"Error loading models: {e}")
    print(f"Looking for models in: {MODEL_DIR}")
    MODELS_LOADED = False

# Custom exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Wine Origin Prediction API",
        "version": "1.0.0",
        "status": "Models loaded" if MODELS_LOADED else "Models not loaded",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/health": "GET - Check API health",
            "/example": "GET - Get example input"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if MODELS_LOADED else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": MODELS_LOADED,
        "model_dir": MODEL_DIR
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_wine_origin(wine: WineInput):
    """
    Predict the country of origin for a wine based on its characteristics
    """
    if not MODELS_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please check server logs."
        )

    try:
        start_time = time.time()

        # Preprocess text features
        X_text = tfidf_vectorizer.transform([wine.description]).toarray()
        
        # Preprocess variety
        try:
            X_variety = variety_encoder.transform([[wine.variety]])
        except ValueError:
            # Handle unknown variety
            X_variety = np.zeros((1, variety_encoder.get_feature_names_out(['variety']).shape[0]))
        
        # Preprocess numerical features
        X_numerical = scaler.transform([[wine.price, wine.points]])
        
        # Combine features
        X = np.hstack([X_text, X_variety, X_numerical])
        
        # Get predictions from both models
        rf_pred_proba = rf_model.predict_proba(X)[0]
        xgb_pred_proba = xgb_model.predict_proba(X)[0]
        
        # Average predictions from both models
        avg_proba = (rf_pred_proba + xgb_pred_proba) / 2
        predicted_idx = np.argmax(avg_proba)
        
        # Get confidence scores
        countries = rf_model.classes_
        confidence_scores = {
            country: float(score) 
            for country, score in zip(countries, avg_proba)
        }
        
        prediction_time = time.time() - start_time
        
        return PredictionResponse(
            predicted_country=countries[predicted_idx],
            confidence_scores=confidence_scores,
            prediction_time=prediction_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/example")
async def get_example():
    """Return an example wine input"""
    return {
        "description": "A rich and full-bodied wine with notes of black cherry and vanilla",
        "points": 92,
        "price": 45.0,
        "variety": "Cabernet Sauvignon"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)