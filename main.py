from dotenv import load_dotenv
from fastapi import FastAPI

from router import router as triage_router

load_dotenv()


def create_app() -> FastAPI:
    app = FastAPI(title="Support Agent")

    @app.get("/health")
    def health_check() -> dict[str, str]:
        return {"status": "ok", "message": "Support agent is running"}

    app.include_router(triage_router)
    return app


app = create_app()
