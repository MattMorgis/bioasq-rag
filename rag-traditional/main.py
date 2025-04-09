from dotenv import load_dotenv
from fastapi import FastAPI
from src.routes import router

# Load environment variables
load_dotenv()

app = FastAPI(title="BioASQ RAG API")

# Include routers
app.include_router(router)
