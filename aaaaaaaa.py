from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
import os
import io

APP_TITLE = "Exoplanet Classifier CSV API (FastAPI)"
MODEL_PATH = os.getenv("MODEL_PATH", "stacking.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "features.json")
TARGET_COL = "koi_disposition"

# Columnas a eliminar del CSV original (tu lista)
DROP_COLS = [
    'rowid','kepid','kepoi_name','kepler_name','koi_pdisposition','koi_score',
    'koi_teq_err1','koi_teq_err2','koi_tce_delivname',
    'koi_fpflag_nt','koi_fpflag_ss','koi_fpflag_co','koi_fpflag_ec'
]

# ------------------------
# Cargar modelo
# ------------------------
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"No pude cargar el modelo desde {MODEL_PATH}: {e}")

# ------------------------
# Resolver columnas esperadas por el modelo
# Prioridad: 1) features.json  2) model.feature_names_in_
# ------------------------
def resolve_feature_cols() -> List[str]:
    # 1) features.json (recomendado)
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            cols = json.load(f)
        if isinstance(cols, list) and all(isinstance(c, str) for c in cols):
            return cols
        raise RuntimeError("features.json inválido (debe ser lista de strings).")

    # 2) Si entrenaste con DataFrame, muchos modelos exponen feature_names_in_
    if hasattr(model, "feature_names_in_"):
        return [str(c) for c in model.feature_names_in_.tolist()]

    raise RuntimeError(
        "No pude determinar las columnas de entrada. Subí un features.json "
        "o guardá el modelo entrenando con DataFrame para tener feature_names_in_."
    )

FEATURE_COLS = resolve_feature_cols()
FEATURE_SET = set(FEATURE_COLS)

# ------------------------
# FastAPI
# ------------------------
app = FastAPI(title=APP_TITLE, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # ajustá a tu front si querés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictCSVResponse(BaseModel):
    n_rows: int
    used_columns: List[str]
    dropped_columns: List[str]
    numeric_columns_detected: List[str]
    predictions: List[int]
    probabilities: Optional[List[List[float]]] = None
    # Matriz de features que se le pasó al modelo (tras dropear y seleccionar num + ordenar)
    model_input_matrix: List[List[float]]

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Subí un CSV al endpoint /predict_csv",
        "n_features_expected": len(FEATURE_COLS),
    }

@app.get("/schema/features")
def schema_features():
    return {"features": FEATURE_COLS}

# ------------------------
# Utilidades de preproceso
# ------------------------
def prepare_dataframe(df: pd.DataFrame):
    # columnas originales
    original_cols = list(df.columns)

    # dropear columnas indicadas y el target si viniera
    drop_now = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_now, errors="ignore")
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL], errors="ignore")

    # seleccionar sólo numéricas
    df_num = df.select_dtypes(include=[np.number])

    # check de columnas requeridas por el modelo
    missing = [c for c in FEATURE_COLS if c not in df_num.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Faltan columnas numéricas requeridas por el modelo: {missing}. "
                f"Numéricas detectadas: {list(df_num.columns)}"
            ),
        )

    # ordenar exactamente como el modelo espera
    X = df_num[FEATURE_COLS].copy()

    # limpiar inf y NaN de ser necesario (si tu pipeline no lo maneja)
    X = X.replace([np.inf, -np.inf], np.nan)
    # Si tu modelo NO es Pipeline con imputer, este fillna evita errores
    try:
        # testeamos rápido si el modelo tiene steps (Pipeline)
        _ = model.named_steps if hasattr(model, "named_steps") else None
        # si no hay pipeline, hacemos fillna con mediana para no romper
        if _ is None:
            X = X.fillna(X.median(numeric_only=True))
    except Exception:
        X = X.fillna(X.median(numeric_only=True))

    return X, original_cols, drop_now, list(df_num.columns)

# ------------------------
# Endpoint principal: subir CSV y predecir
# return_features: si "true", devolvemos la matriz que se pasó al modelo (model_input_matrix)
# ------------------------
@app.post("/predict_csv", response_model=PredictCSVResponse)
async def predict_csv(
    file: UploadFile = File(..., description="CSV en formato original Kepler/KOI"),
):
    # Leer archivo subido
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No pude leer el CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="El CSV está vacío.")

    # Preprocesar: dropear columnas, seleccionar num, ordenar
    X, original_cols, dropped_cols, numeric_cols_detected = prepare_dataframe(df)

    # Armar matriz de entrada para el modelo
    X_np = X.to_numpy(dtype=float)

    # Predecir
    try:
        yhat = model.predict(X_np).astype(int).tolist()
        proba = model.predict_proba(X_np).tolist() if hasattr(model, "predict_proba") else None
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error durante la predicción: {e}")

    # Devolver también la matriz de entrada (lo que “queda” tras tus reglas)
    return PredictCSVResponse(
        n_rows=len(X),
        used_columns=FEATURE_COLS,
        dropped_columns=dropped_cols,
        numeric_columns_detected=numeric_cols_detected,
        predictions=yhat,
        probabilities=proba,
        model_input_matrix=X_np.tolist(),
    )