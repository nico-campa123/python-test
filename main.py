from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import sklearn
import pandas as pd


app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace * with your frontend domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


df=pd.read_csv("cumulative_2025.10.04_05.21.55.csv")
model=joblib.load("stackin.pkl")



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