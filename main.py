from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import csv
import io
from typing import Optional

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace * with your frontend domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"message": "World"}


@app.get("/deploy")
async def deploy():
    return {"message": "Deploy endpoint"}


def _is_likely_csv(filename: Optional[str], content_type: Optional[str]) -> bool:
    if content_type:
        ct = content_type.lower()
        if "csv" in ct or "comma-separated-values" in ct:
            return True
    if filename:
        if filename.lower().endswith('.csv'):
            return True
    return False


@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    # Basic checks to ensure it's likely a CSV
    if not _is_likely_csv(getattr(file, 'filename', None), getattr(file, 'content_type', None)):
        raise HTTPException(status_code=400, detail="Uploaded file does not appear to be a CSV")

    content = await file.read()
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError:
        # Try latin-1 as fallback; if that fails, report error
        try:
            text = content.decode('latin-1')
        except Exception:
            raise HTTPException(status_code=400, detail="Unable to decode uploaded file as text")

    # Parse CSV and count rows/columns
    reader = csv.reader(io.StringIO(text))
    rows = []
    for row in reader:
        rows.append(row)

    if not rows:
        raise HTTPException(status_code=400, detail="CSV appears to be empty")

    num_rows = len(rows)
    num_columns = max(len(r) for r in rows)

    # Return a small summary (don't echo entire file)
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "rows": num_rows,
        "columns": num_columns,
        "first_row": rows[0],
    }