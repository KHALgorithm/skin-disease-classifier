import warnings

from app.api.routes import router
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

warnings.filterwarnings("ignore")

app = FastAPI(
    title="Skin Diseases Classification API",
    description="""An API that utilises a Deep Learning model built with Keras(Tensorflow) to detect skin diseases from images""",
    version="0.0.1",
    debug=True,
)


@app.get("/", response_class=PlainTextResponse)
async def running():
    note = """
Skin Disease Classification API 🙌🏻

Welcome to the Skin Disease Classification API!
Note: add "/docs" to the URL to get the Swagger UI Docs or "/redoc"
"""
    return note


app.include_router(router)

# python -m uvicorn app.main:app --port 5353 --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5353)
