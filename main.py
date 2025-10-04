from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib, sklearn, os, sys
import pandas as pd

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE_NAME = "stackin.pkl"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILE_NAME)

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from: {MODEL_PATH}", file=sys.stderr) 

except FileNotFoundError:
    print(f"FATAL ERROR: Model file not found at {MODEL_PATH}", file=sys.stderr)
    raise


model=joblib.load(MODEL_FILE_NAME)



@app.get("/")
async def read_root():
    return {"message": "World"}


@app.get("/deploy")
async def deploy():
    return {"message": "Deploy endpoint"}


@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()
    return {"characters": len(content.decode("utf-8"))}