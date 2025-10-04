from fastapi import FastAPI

app = FastAPI()
@app.get("/")
async def read_root():
    return {"message": "World"}
@app.get("/deploy")
async def deploy():
    return {"message": "Deploy endpoint"}