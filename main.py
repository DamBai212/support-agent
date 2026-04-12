from fastapi import FastAPI
from dotenv import load_dotenv
import os

load_dotenv()  # reads my .env file

app = FastAPI(title="Support Agent")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Support agent is running"}