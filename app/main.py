from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from app import config  # Load local environment before importing services.
from app.api.routes_lessons import router as lessons_router
from app.api.routes_saved_lessons import router as saved_lessons_router
from app.database import migrate_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(migrate_database)
    yield

app = FastAPI(
    title="English Lesson App Backend",
    version="1.1.0",
    lifespan=lifespan,
)


@app.exception_handler(SQLAlchemyError)
async def database_error(request: Request, error: SQLAlchemyError):
    logging.getLogger(__name__).error("Database operation failed (%s)", type(error).__name__)
    return JSONResponse(status_code=503, content={"detail": "The lesson database is unavailable. Please try again later."})


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
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Content-Type"],
    expose_headers=["Content-Disposition"],
)

app.include_router(lessons_router)
app.include_router(saved_lessons_router)
