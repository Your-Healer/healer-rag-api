# main.py
import platform
import asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from starlette.responses import JSONResponse

from app.config import VectorDBType, debug_mode, RAG_HOST, RAG_PORT, CHUNK_SIZE, CHUNK_OVERLAP, PDF_EXTRACT_IMAGES, VECTOR_DB_TYPE, \
    LogMiddleware, logger
from app.middleware import security_middleware
from app.routes import langchain_routes

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic goes here
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    yield

app = FastAPI(lifespan=lifespan, debug=debug_mode)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(LogMiddleware)

app.middleware("http")(security_middleware)

# Set state variables for use in routes
app.state.CHUNK_SIZE = CHUNK_SIZE
app.state.CHUNK_OVERLAP = CHUNK_OVERLAP
app.state.PDF_EXTRACT_IMAGES = PDF_EXTRACT_IMAGES

# Include routers
app.include_router(router=langchain_routes.router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logger.debug(f"Validation error occurred")
    logger.debug(f"Raw request body: {body.decode()}")
    logger.debug(f"Validation errors: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": body.decode(),
            "message": "Request validation failed",
        },
    )

if __name__ == "__main__":
    uvicorn.run(app, host=RAG_HOST, port=RAG_PORT, log_config=None)