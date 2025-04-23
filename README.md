# README.md

# Skin Disease Classifier

A FastAPI application that classifies skin diseases into three categories (Infectious, Benign, Malignant) using a VGG19-based deep learning model implemented with TensorFlow/Keras.

## Project Structure

```
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── api/
│   │   ├── routes.py        # API endpoint definitions
│   │   └── schemas.py       # Pydantic data models
│   ├── core/
│   │   ├── config.py        # Application configuration
│   │   └── logger.py        # Logging setup
│   └── src/
│       ├── classifier.py     # ML model wrapper
│       └── preprocessing.py  # Image preprocessing utilities
├── docker-compose.yml       # Docker composition config
├── Dockerfile              # Container definition
└── requirements.txt        # Python dependencies
```

### Using Docker (Recommended)

1.  Build and run the application using Docker Compose:

    ```bash
    docker-compose up --build
    ```

### Manual Setup

1.  Create a Python virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  Run the application:

    ```bash
    python -m uvicorn app.main:app --port 5353 --reload
    ```

    *Note: The `--reload` flag enables automatic reloading upon code changes, which is useful during development.*

## API Endpoints

*   `GET /` - Welcome message and API information.
*   `POST /classify` - Upload and classify a skin disease image.
    *   Accepts image files (JPEG, PNG).
    *   Returns prediction label and confidence score.

### Example Response

```json
{
    "prediction": "Benign",
    "confidence": 0.95
}

## Disease Categories

The model classifies images into three categories:
- 1. Enfeksiyonel (Infectious)
- 5. Benign
- 6. Malign

## License

This project is licensed under the MIT License.
