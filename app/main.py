import asyncio
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from app.config import AppSettings
from app.api.routes_lessons import router as lessons_router
from app.api.routes_saved_lessons import router as saved_lessons_router
from app.database import get_engine, migrate_database
from app.sharing import FrontendFiles, ShareProtectionMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(migrate_database)
    try:
        yield
    finally:
        await run_in_threadpool(get_engine().dispose)

def create_app(settings: AppSettings | None = None) -> FastAPI:
    settings = settings or AppSettings.from_environment()
    settings.validate()
    application = FastAPI(
        title="English Lesson App Backend", version="1.1.0", lifespan=lifespan,
    )
    application.state.generation_slots = asyncio.Semaphore(1)

    @application.exception_handler(SQLAlchemyError)
    async def database_error(request: Request, error: SQLAlchemyError):
        logging.getLogger(__name__).error("Database operation failed (%s)", type(error).__name__)
        return JSONResponse(status_code=503, content={"detail": "The lesson database is unavailable. Please try again later."})

    @application.get("/")
    async def root():
        if settings.frontend_dist:
            return FileResponse(settings.frontend_dist / "index.html", headers={"Cache-Control": "no-cache"})
        return {"message": "Backend dziala", "frontend_dev_url": "http://127.0.0.1:5173"}

    @application.get("/health")
    @application.get("/api/health", include_in_schema=False)
    async def health():
        return {"message": "Backend dziala"}

    @application.get("/api/openapi.json", include_in_schema=False)
    async def api_schema():
        return application.openapi()

    application.add_middleware(ShareProtectionMiddleware, settings=settings)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins) if not settings.public_share else [],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Content-Type"], expose_headers=["Content-Disposition"],
    )
    for prefix in ("", "/api"):
        application.include_router(lessons_router, prefix=prefix)
        application.include_router(saved_lessons_router, prefix=prefix)
    if settings.frontend_dist:
        application.mount("/", FrontendFiles(directory=settings.frontend_dist, html=True), name="frontend")
    return application


app = create_app()
