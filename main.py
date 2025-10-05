from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib, os, sys
import pandas as pd

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE_NAME = "stacking.pkl"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILE_NAME)

# Load model once
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from: {MODEL_PATH}", file=sys.stderr)
except FileNotFoundError:
    print(f"❌ ERROR: Model file not found at {MODEL_PATH}", file=sys.stderr)
    raise


@app.get("/")
async def root():
    return {"message": "Hello from FastAPI on Render!"}


@app.get("/deploy")
async def deploy():
    return {"message": "Deploy endpoint working!"}


@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(pd.compat.StringIO(content.decode("utf-8")))
        return {"rows": df.shape[0], "columns": df.shape[1]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")
