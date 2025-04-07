import cv2
import numpy as np
import io
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from app.api.schemas import ClassificationResponse
from app.src.classifier import Classifier
from app.src.preprocessing import preprocess_image
from app.core.logger import logger

logger = logger(__name__)
router = APIRouter()


@router.post("/classify", response_model=ClassificationResponse)
async def classify_skin_disease(image: UploadFile = File(
        ..., description="Document image file (JPEG, PNG, or TIFF)"
)):
    try:
        logger.info(f"Processing new image classification request: {image.filename}")

        # Read the image file into memory
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.error(f"Failed to decode image: {image.filename}")
            raise ValueError("Failed to decode image")

        # Use the preprocessing function to prepare the image
        logger.debug("Preprocessing image...")
        processed_img = preprocess_image(img)
        logger.debug("Image preprocessing completed")

        # Initialize the classifier
        logger.debug("Initializing classifier...")
        classifier = Classifier()

        # Make prediction using preprocessed image
        logger.debug("Making prediction...")
        label, confidence = classifier.predict(processed_img)
        logger.info(f"Classification completed - Label: {label}, Confidence: {confidence:.2f}")

        return JSONResponse(content={
            "prediction": label,
            "confidence": float(confidence)
        })
    except Exception as e:
        logger.error(f"Error in classification for {image.filename}: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"message": f"Classification error: {str(e)}"}
        )
