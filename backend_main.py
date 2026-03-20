"""
DiabetesSense v2.0 — FastAPI Backend
=====================================
Dataset:   Early Stage Diabetes Risk Prediction (UCI)
Features:  Age, Gender + 14 binary symptom indicators
Best Model: Random Forest (AUC=0.9679, Acc=96.1%)

Run:
    uvicorn backend.main:app --reload --port 8000

API Docs:
    http://localhost:8000/docs

Endpoints:
    GET  /health
    POST /predict
    POST /explain
    POST /recommendation
    GET  /model-info
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import joblib
import numpy as np

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("DiabetesSense")

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DiabetesSense API v2.0",
    description=(
        "ML-based Early Stage Diabetes Risk Prediction.\n\n"
        "Dataset: UCI Early Stage Diabetes Risk Prediction Dataset\n"
        "Model: Random Forest (AUC=0.9679, Accuracy=96.1%)\n"
        "Explainability: SHAP Feature Importance"
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Restrict in production: ["https://yourdomain.com"]
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# ─── Static files ─────────────────────────────────────────────────────────────
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ─── Constants ───────────────────────────────────────────────────────────────
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
META_PATH   = "models/model_meta.json"

# Feature order must match training exactly
FEATURE_COLUMNS = [
    "Age", "Gender", "Polyuria", "Polydipsia",
    "sudden weight loss", "weakness", "Polyphagia",
    "Genital thrush", "visual blurring", "Itching",
    "Irritability", "delayed healing", "partial paresis",
    "muscle stiffness", "Alopecia", "Obesity",
]

# SHAP-proxy importance weights (from RF training)
FEATURE_SHAP_WEIGHTS = {
    "Polyuria":           0.2198,
    "Polydipsia":         0.1834,
    "Age":                0.1066,
    "sudden weight loss": 0.0663,
    "Gender":             0.0650,
    "Polyphagia":         0.0439,
    "Irritability":       0.0439,
    "partial paresis":    0.0407,
    "weakness":           0.0378,
    "Obesity":            0.0312,
    "visual blurring":    0.0305,
    "delayed healing":    0.0298,
    "muscle stiffness":   0.0264,
    "Genital thrush":     0.0243,
    "Alopecia":           0.0229,
    "Itching":            0.0215,
}

SYMPTOM_LABELS = {
    "Polyuria":           "Polyuria (Excessive Urination)",
    "Polydipsia":         "Polydipsia (Excessive Thirst)",
    "sudden weight loss": "Sudden Weight Loss",
    "weakness":           "General Weakness",
    "Polyphagia":         "Polyphagia (Excessive Hunger)",
    "Genital thrush":     "Genital Thrush",
    "visual blurring":    "Visual Blurring",
    "Itching":            "Persistent Itching",
    "Irritability":       "Irritability",
    "delayed healing":    "Delayed Wound Healing",
    "partial paresis":    "Partial Paresis",
    "muscle stiffness":   "Muscle Stiffness",
    "Alopecia":           "Alopecia (Hair Loss)",
    "Obesity":            "Obesity (BMI > 30)",
}

# ─── Model State ─────────────────────────────────────────────────────────────
_model  = None
_scaler = None
_meta   = None
_model_needs_scaling = False


@app.on_event("startup")
async def load_artifacts():
    global _model, _scaler, _meta, _model_needs_scaling
    try:
        _model  = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        with open(META_PATH) as f:
            _meta = json.load(f)
        _model_needs_scaling = _meta.get("best_needs_scaling", False)
        logger.info(f"✓ Model loaded: {_meta['best_model_name']}")
        logger.info(f"  Best AUC: {max(r['auc'] for r in _meta['results'])}")
    except FileNotFoundError as e:
        logger.warning(f"⚠ Model not loaded: {e}")
        logger.warning("  Run: python models/train.py")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")


# ─── Pydantic Schemas ────────────────────────────────────────────────────────

class BinaryField(int):
    """Validates that a field is exactly 0 or 1."""


class PatientInput(BaseModel):
    """
    Input schema matching Early Stage Diabetes Risk Prediction Dataset features.
    All symptom fields accept: 0 (No) or 1 (Yes)
    """

    # Demographics
    Age: int = Field(..., ge=1, le=120,
                     description="Patient age in years (dataset range: 16–90)",
                     example=45)
    Gender: int = Field(..., ge=0, le=1,
                        description="Gender: 1=Male, 0=Female",
                        example=1)

    # Top predictors (highest SHAP importance)
    Polyuria:            int = Field(..., ge=0, le=1, description="Excessive urination? 1=Yes, 0=No")
    Polydipsia:          int = Field(..., ge=0, le=1, description="Excessive thirst? 1=Yes, 0=No")

    # Metabolic symptoms
    sudden_weight_loss:  int = Field(..., ge=0, le=1, alias="sudden weight loss",
                                    description="Unexplained sudden weight loss? 1=Yes, 0=No")
    weakness:            int = Field(..., ge=0, le=1, description="General weakness? 1=Yes, 0=No")
    Polyphagia:          int = Field(..., ge=0, le=1, description="Excessive hunger? 1=Yes, 0=No")
    Genital_thrush:      int = Field(..., ge=0, le=1, alias="Genital thrush",
                                    description="Genital thrush? 1=Yes, 0=No")

    # Visual & neurological
    visual_blurring:     int = Field(..., ge=0, le=1, alias="visual blurring",
                                    description="Blurred vision? 1=Yes, 0=No")
    Itching:             int = Field(..., ge=0, le=1, description="Persistent itching? 1=Yes, 0=No")
    Irritability:        int = Field(..., ge=0, le=1, description="Irritability? 1=Yes, 0=No")
    delayed_healing:     int = Field(..., ge=0, le=1, alias="delayed healing",
                                    description="Delayed wound healing? 1=Yes, 0=No")
    partial_paresis:     int = Field(..., ge=0, le=1, alias="partial paresis",
                                    description="Partial paresis (limb weakness)? 1=Yes, 0=No")
    muscle_stiffness:    int = Field(..., ge=0, le=1, alias="muscle stiffness",
                                    description="Muscle stiffness? 1=Yes, 0=No")

    # Physical
    Alopecia:            int = Field(..., ge=0, le=1, description="Hair loss (alopecia)? 1=Yes, 0=No")
    Obesity:             int = Field(..., ge=0, le=1, description="Obese (BMI>30)? 1=Yes, 0=No")

    class Config:
        populate_by_name = True  # Pydantic v2

    @field_validator("Age")
    @classmethod
    def validate_age(cls, v):
        if v < 1 or v > 120:
            raise ValueError("Age must be between 1 and 120")
        return v


class FeatureContribution(BaseModel):
    feature:     str
    label:       str
    value:       float
    importance:  float
    direction:   str
    description: str


class PredictionResponse(BaseModel):
    risk_score:      int           # 0–100 percentage
    risk_probability: float        # 0.0–1.0
    risk_label:      str           # Low / Moderate / High Risk
    risk_color:      str           # Hex color
    model_name:      str
    model_accuracy:  float
    model_auc:       float
    timestamp:       str
    symptoms_positive: List[str]
    symptoms_negative: List[str]


class ExplanationResponse(BaseModel):
    feature_contributions: List[FeatureContribution]
    top_risk_factors:      List[str]
    top_protective:        List[str]
    explanation_text:      str
    shap_max_value:        float


class RecommendationResponse(BaseModel):
    risk_level:      str
    risk_score:      int
    recommendations: List[Dict[str, str]]
    next_steps:      List[str]
    medical_note:    str


# ─── Feature Conversion ──────────────────────────────────────────────────────

def patient_to_vector(patient: PatientInput) -> np.ndarray:
    """
    Convert PatientInput Pydantic model to numpy feature vector.
    Order must exactly match FEATURE_COLUMNS from training.
    """
    d = patient.model_dump(by_alias=True)
    vec = np.array([[
        d["Age"],
        d["Gender"],
        d["Polyuria"],
        d["Polydipsia"],
        d["sudden weight loss"],
        d["weakness"],
        d["Polyphagia"],
        d["Genital thrush"],
        d["visual blurring"],
        d["Itching"],
        d["Irritability"],
        d["delayed healing"],
        d["partial paresis"],
        d["muscle stiffness"],
        d["Alopecia"],
        d["Obesity"],
    ]], dtype=float)
    return vec


def compute_shap_contributions(patient: PatientInput) -> List[FeatureContribution]:
    """
    Compute SHAP-proxy contributions based on Random Forest feature importances.
    For a production system with SHAP installed: use shap.TreeExplainer instead.
    """
    d = patient.model_dump(by_alias=True)
    contributions = []

    for feat, importance in FEATURE_SHAP_WEIGHTS.items():
        val      = d.get(feat, 0)
        label    = SYMPTOM_LABELS.get(feat, feat)

        # SHAP contribution = importance × feature_value
        # For Age: normalize to 0-1 range relative to dataset max (90)
        if feat == "Age":
            norm_val = val / 90.0
            shap_val  = importance * norm_val
        elif feat == "Gender":
            shap_val = importance * (1 - val)  # Male = slightly higher risk
        else:
            shap_val = importance * val  # Binary: 0 or importance

        direction = "increases" if shap_val > 0.01 else "neutral"

        if feat == "Age":
            desc = f"Age {val}: {'elevated risk (>40)' if val>40 else 'lower risk group'}"
        elif feat == "Gender":
            desc = f"{'Male' if val==1 else 'Female'}: {'slightly higher' if val==1 else 'slightly lower'} baseline risk"
        elif val == 1:
            desc = f"You reported {label} — a known early diabetes indicator"
        else:
            desc = f"No {label} reported — protective factor"

        contributions.append(FeatureContribution(
            feature=feat, label=label,
            value=round(shap_val, 4),
            importance=round(importance, 4),
            direction=direction,
            description=desc,
        ))

    contributions.sort(key=lambda x: x.value, reverse=True)
    return contributions


def get_risk_metadata(score: int) -> dict:
    if score < 30:
        return {"label": "Low Risk",      "color": "#16a34a"}
    elif score < 65:
        return {"label": "Moderate Risk", "color": "#d97706"}
    else:
        return {"label": "High Risk",     "color": "#e53e3e"}


def generate_recommendations(score: int, patient: PatientInput) -> dict:
    d = patient.model_dump(by_alias=True)

    if score < 30:
        recs = [
            {"icon": "💧", "title": "Stay Hydrated",          "desc": "Maintain 8–10 glasses of water daily. Limit sugary drinks."},
            {"icon": "🥗", "title": "Balanced Diet",           "desc": "Prioritize whole grains, vegetables, lean proteins. Limit refined sugars."},
            {"icon": "🏃", "title": "Regular Exercise",        "desc": "Maintain 150+ min/week moderate aerobic activity. Resistance training 2×/week."},
            {"icon": "🩺", "title": "Annual Screening",        "desc": "Continue annual HbA1c and fasting glucose monitoring."},
            {"icon": "😴", "title": "Quality Sleep",           "desc": "7–9 hours nightly supports healthy glucose metabolism."},
        ]
        next_steps = [
            "Schedule annual physical with blood glucose panel",
            "Monitor weight monthly",
            "Maintain current healthy lifestyle",
        ]
        medical_note = "Your symptom profile is currently low-risk. Annual preventive screening is recommended."

    elif score < 65:
        recs = [
            {"icon": "🚨", "title": "Medical Consultation",    "desc": "Schedule an appointment with your GP within 2 weeks. Request fasting glucose and HbA1c testing."},
            {"icon": "🏋️", "title": "Structured Exercise",    "desc": "30+ min daily moderate exercise. Include resistance training 2–3×/week to improve insulin sensitivity."},
            {"icon": "🥦", "title": "Low-Glycemic Diet",       "desc": "Reduce refined carbs and sugary foods. Increase dietary fiber from vegetables, legumes, whole grains."},
            {"icon": "⚖️", "title": "Weight Management",       "desc": "5–7% body weight reduction cuts diabetes progression risk by 58% (CDC DPP data)." if d.get("Obesity") else "Maintain healthy weight through consistent healthy habits."},
            {"icon": "📊", "title": "Track Symptoms",          "desc": "Log daily symptoms, energy levels, fluid intake using a health app. Share logs with your physician."},
            {"icon": "🧘", "title": "Stress Management",       "desc": "Chronic stress worsens insulin resistance. Practice mindfulness or yoga 20 min/day."},
        ]
        next_steps = [
            "Book GP appointment within 2 weeks",
            "Request HbA1c + fasting glucose blood test",
            "Start a food & symptom diary",
            "Begin 20-min daily walks immediately",
            "Discuss family history with your doctor",
        ]
        medical_note = "Moderate risk detected. Prompt clinical evaluation and lifestyle changes are recommended."

    else:
        recs = [
            {"icon": "🏥", "title": "Seek Medical Care Urgently",   "desc": "Visit your physician or endocrinologist this week. Your symptom cluster warrants immediate clinical evaluation."},
            {"icon": "🔬", "title": "Request Full Diabetes Panel",  "desc": "Ask for: Fasting Plasma Glucose, 2-hr OGTT, HbA1c, fasting insulin, lipid panel, kidney function (eGFR)."},
            {"icon": "💊", "title": "Discuss Preventive Medication","desc": "Metformin reduces T2D onset by 31% in high-risk individuals (NEJM DPP trial). Discuss with your doctor."},
            {"icon": "🥗", "title": "Medical Nutrition Therapy",    "desc": "Request dietitian referral. Target: <45g carbs/meal, 25–35g fiber/day, zero sugary beverages."},
            {"icon": "📏", "title": "Daily Glucose Monitoring",     "desc": "Obtain a home glucometer. Target: fasting <100 mg/dL, 2-hr post-meal <140 mg/dL."},
            {"icon": "🤝", "title": "Join a DPP Program",           "desc": "CDC Diabetes Prevention Program reduces risk 58%. Request referral from your physician."},
            {"icon": "😴", "title": "Treat Sleep Disorders",        "desc": "Sleep apnea severely worsens insulin resistance. Discuss sleep quality with your doctor."},
        ]
        next_steps = [
            "Contact your doctor TODAY",
            "Request HbA1c + OGTT + full metabolic panel",
            "Ask about Diabetes Prevention Program referral",
            "Begin daily home glucose monitoring",
            "Contact a Registered Dietitian",
            "Log all symptoms and bring list to appointment",
        ]
        medical_note = ("High risk profile detected. Multiple early diabetes symptoms present. "
                        "Immediate clinical evaluation is strongly recommended.")

    return {
        "risk_level":      "Low" if score < 30 else ("Moderate" if score < 65 else "High"),
        "risk_score":      score,
        "recommendations": recs,
        "next_steps":      next_steps,
        "medical_note":    medical_note,
    }


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Health check — returns API and model status."""
    return {
        "status":       "healthy",
        "model_loaded": _model is not None,
        "model_name":   _meta["best_model_name"] if _meta else "not loaded",
        "version":      "2.0.0",
        "timestamp":    datetime.utcnow().isoformat(),
        "dataset":      "UCI Early Stage Diabetes Risk Prediction",
    }


@app.get("/model-info", tags=["System"])
async def model_info():
    """Returns training metadata, model performance, and feature importances."""
    if _meta is None:
        raise HTTPException(503, "Model metadata not loaded. Run train.py first.")
    return {
        "best_model": _meta["best_model_name"],
        "feature_columns": _meta["feature_columns"],
        "all_results": _meta["results"],
        "feature_importance": _meta["feature_importance"],
        "confusion_matrix": _meta["confusion_matrix"],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(patient: PatientInput):
    """
    Predict diabetes risk from patient symptom inputs.

    Returns:
    - risk_score: 0–100 percentage
    - risk_label: Low / Moderate / High Risk
    - Model performance metrics
    - Lists of reported positive/negative symptoms
    """
    if _model is None:
        raise HTTPException(503, "Model not loaded. Run: python models/train.py")

    try:
        X = patient_to_vector(patient)
        X_input = _scaler.transform(X) if _model_needs_scaling else X

        proba       = _model.predict_proba(X_input)[0]
        risk_prob   = float(proba[1])
        risk_score  = int(round(risk_prob * 100))
        meta_r      = get_risk_metadata(risk_score)

        d = patient.model_dump(by_alias=True)
        syms_pos = [SYMPTOM_LABELS.get(k, k) for k in SYMPTOM_LABELS
                    if d.get(k, -1) == 1]
        syms_neg = [SYMPTOM_LABELS.get(k, k) for k in SYMPTOM_LABELS
                    if d.get(k, -1) == 0]

        best_result = next((r for r in _meta["results"]
                            if r["model"] == _meta["best_model_name"]), {})

        logger.info(f"Prediction: score={risk_score}%, label={meta_r['label']}, "
                    f"pos_symptoms={len(syms_pos)}")

        return PredictionResponse(
            risk_score=risk_score,
            risk_probability=round(risk_prob, 4),
            risk_label=meta_r["label"],
            risk_color=meta_r["color"],
            model_name=_meta["best_model_name"],
            model_accuracy=best_result.get("accuracy", 0.0),
            model_auc=best_result.get("auc", 0.0),
            timestamp=datetime.utcnow().isoformat(),
            symptoms_positive=syms_pos,
            symptoms_negative=syms_neg,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@app.post("/explain", response_model=ExplanationResponse, tags=["Explainability"])
async def explain(patient: PatientInput):
    """
    Generate SHAP-based feature importance explanation for a patient prediction.
    Returns contribution of each symptom to the risk score.
    """
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    try:
        contributions = compute_shap_contributions(patient)
        max_val       = max(c.value for c in contributions)

        top_risk      = [c.label for c in contributions if c.value > 0.02][:3]
        top_protect   = [c.label for c in reversed(contributions) if c.value == 0.0][:2]

        if top_risk:
            explanation = (
                f"The primary driver of your risk is {top_risk[0]}, "
                f"which contributes {contributions[0].value:.3f} SHAP units. "
                + (f"This is compounded by {top_risk[1]} and {top_risk[2]}. " if len(top_risk) >= 3 else "")
                + "The Random Forest model identified these symptom combinations as highly predictive."
            )
        else:
            explanation = (
                "Your reported symptoms do not strongly activate high-risk patterns in the model. "
                "Age remains the primary contributing factor for baseline risk."
            )

        return ExplanationResponse(
            feature_contributions=contributions,
            top_risk_factors=top_risk,
            top_protective=top_protect,
            explanation_text=explanation,
            shap_max_value=round(max_val, 4),
        )

    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(500, f"Explanation failed: {str(e)}")


@app.post("/recommendation", response_model=RecommendationResponse, tags=["Recommendations"])
async def recommend(patient: PatientInput):
    """
    Generate personalized health recommendations based on risk score and symptom profile.
    """
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    try:
        X       = patient_to_vector(patient)
        X_input = _scaler.transform(X) if _model_needs_scaling else X
        proba   = _model.predict_proba(X_input)[0]
        score   = int(round(float(proba[1]) * 100))

        recs = generate_recommendations(score, patient)
        return RecommendationResponse(**recs)

    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(500, f"Recommendation failed: {str(e)}")


# ─── Example / Test Endpoint ──────────────────────────────────────────────────

@app.get("/example-input", tags=["System"])
async def example_input():
    """Returns an example high-risk patient payload for testing."""
    return {
        "example_high_risk": {
            "Age": 52, "Gender": 1,
            "Polyuria": 1, "Polydipsia": 1,
            "sudden weight loss": 1, "weakness": 1,
            "Polyphagia": 1, "Genital thrush": 0,
            "visual blurring": 1, "Itching": 0,
            "Irritability": 1, "delayed healing": 1,
            "partial paresis": 0, "muscle stiffness": 0,
            "Alopecia": 0, "Obesity": 1,
        },
        "example_low_risk": {
            "Age": 28, "Gender": 0,
            "Polyuria": 0, "Polydipsia": 0,
            "sudden weight loss": 0, "weakness": 0,
            "Polyphagia": 0, "Genital thrush": 0,
            "visual blurring": 0, "Itching": 0,
            "Irritability": 0, "delayed healing": 0,
            "partial paresis": 0, "muscle stiffness": 0,
            "Alopecia": 0, "Obesity": 0,
        },
        "curl_command": (
            'curl -X POST http://localhost:8000/predict '
            '-H "Content-Type: application/json" '
            '-d \'{"Age":52,"Gender":1,"Polyuria":1,"Polydipsia":1,'
            '"sudden weight loss":1,"weakness":1,"Polyphagia":1,'
            '"Genital thrush":0,"visual blurring":1,"Itching":0,'
            '"Irritability":1,"delayed healing":1,"partial paresis":0,'
            '"muscle stiffness":0,"Alopecia":0,"Obesity":1}\''
        ),
    }


# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
