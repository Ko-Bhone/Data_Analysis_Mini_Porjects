from fastapi import FastAPI

app = FastAPI(
    title="Employee Analysis API",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Employee Analysis API is Running..."}
