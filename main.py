from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("Stacking_fp.pkl")
path = "cumulative_2025.10.04_05.21.55.csv"
df = pd.read_csv(
    path,
    comment='#',      # ignore metadata lines starting with '#'
    engine='python',  # more forgiving parser
)

def que_pruebe(content):
    try:
        df_init = pd.read_csv(StringIO(content.decode("utf-8")))
    except Exception:
        try:
            df_init = pd.read_csv(StringIO(content.decode("latin-1")))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error parsing CSV: {e}")

    # capture kepid before we drop columns
    kepids = df_init['kepid'].tolist() if 'kepid' in df_init.columns else [None] * len(df_init)

    # Drop unwanted columns
    drop = [
        'rowid','kepid','kepoi_name','kepler_name','koi_pdisposition',
        'koi_score','koi_teq_err1','koi_teq_err2','koi_tce_delivname',
        'koi_fpflag_nt','koi_fpflag_ss','koi_fpflag_co','koi_fpflag_ec'
    ]
    df = df_init.drop(columns=[c for c in drop if c in df_init.columns], errors="ignore")

    # Replace NaN with median
    df = df.fillna(df.median(numeric_only=True))

    # Split features
    if 'koi_disposition' in df.columns:
        df = df.drop(columns=['koi_disposition'], errors='ignore')
    if 'target' in df.columns:
        df = df.drop(columns=['target'], errors='ignore')

    # Drop non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    X = df.drop(columns=non_numeric_cols)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Make predictions
    preds = model.predict(X_scaled)
    preds_list = [int(p) for p in preds]
    paired = [[kepids[i], preds_list[i]] for i in range(min(len(kepids), len(preds_list)))]
    return {"predictions": paired}

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI on Render!"}

@app.get("/deploy")
async def deploy():
    return {"message": "Deploy endpoint working!"}

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    # Read CSV
    content = await file.read()
    try:
        df_init = pd.read_csv(StringIO(content.decode("utf-8")))
    except Exception:
        try:
            df_init = pd.read_csv(StringIO(content.decode("latin-1")))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error parsing CSV: {e}")

    # capture kepid before we drop columns
    kepids = df_init['kepid'].tolist() if 'kepid' in df_init.columns else [None] * len(df_init)

    # Drop unwanted columns
    drop = [
        'rowid','kepid','kepoi_name','kepler_name','koi_pdisposition',
        'koi_score','koi_teq_err1','koi_teq_err2','koi_tce_delivname',
        'koi_fpflag_nt','koi_fpflag_ss','koi_fpflag_co','koi_fpflag_ec'
    ]
    df = df_init.drop(columns=[c for c in drop if c in df_init.columns], errors="ignore")

    # Replace NaN with median
    df = df.fillna(df.median(numeric_only=True))

    # Split features
    if 'koi_disposition' in df.columns:
        df = df.drop(columns=['koi_disposition'], errors='ignore')
    if 'target' in df.columns:
        df = df.drop(columns=['target'], errors='ignore')

    # Drop non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    X = df.drop(columns=non_numeric_cols)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Make predictions
    preds = model.predict(X_scaled)
    preds_list = [int(p) for p in preds]
    paired = [[kepids[i], preds_list[i]] for i in range(min(len(kepids), len(preds_list)))]
    return {"predictions": paired}

@app.get("/campa")
async def tr():
    kepler = df.drop(columns=['koi_disposition'], errors='ignore')
    non_numeric_cols = kepler.select_dtypes(exclude=np.number).columns.tolist()
    X = kepler.drop(columns=non_numeric_cols)
    X= StandardScaler().fit_transform(X)
    a = model.predict(X)
    return {"message": "Hello from FastAPI on Render!", "predictions": a}