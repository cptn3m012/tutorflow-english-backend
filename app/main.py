from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_lessons import router as lessons_router

app = FastAPI(
    title="English Lesson App Backend",
    version="1.0.0"
)


@app.get("/")
async def root():
    return {"message": "Backend dziala", "frontend_dev_url": "http://127.0.0.1:5173"}


@app.get("/health")
async def health():
    return {"message": "Backend dziala"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    expose_headers=["Content-Disposition"],
)

app.include_router(lessons_router)
