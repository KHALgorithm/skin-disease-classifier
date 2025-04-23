import os
import threading

import numpy as np
from tensorflow.keras.models import load_model

from app.core.logger import logger
from app.src.preprocessing import (
    preprocess_image,
)

logger = logger(__name__)

# Define class labels
CLASS_LABELS = ["Enfeksiyonel", "Benign", "Malign"]


class Classifier:
    """
    A service class to manage machine learning model loading and predictions.
    Ensures thread-safe operations and provides reusable methods for interacting with the model.
    """

    # Lock for the model in thread to ensure thread safety
    _model_lock = threading.Lock()

    def __init__(self, model_path="model/bestmodel_98.keras"):
        """
        Initializes the Classifier and loads the model in a thread-safe manner.

        Args:
            model_path (str): The file path to the saved model. Defaults to "bestmodel_93.keras"
                             relative to the script's directory.
        """
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """
        Loads the machine learning model in a thread-safe manner.
        Ensures the model is loaded only once, even in multi-threaded environments.
        """
        # Open Thread lock to load the model
        with self._model_lock:
            if self.model is None:
                try:
                    logger.info(f"Loading ML model from {self.model_path}...")
                    # Load the model
                    # Update model path to be relative to script location
                    model_path = os.path.join(
                        os.path.dirname(__file__), self.model_path
                    )
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"Model file not found at {model_path}")

                    self.model = load_model(model_path)
                    logger.info("ML model loaded successfully.")
                except Exception as e:
                    logger.error(f"Error loading ML model: {e}")
                    raise

    def predict(self, image):
        """
        Predict skin disease from image using the loaded trained model.

        Args:
            image: Preprocessed image array of shape (height, width, channels)

        Returns:
            tuple: (prediction_label, prediction_index)
                prediction_label: String name of predicted disease class
                prediction_index: Integer index of predicted class

        Raises:
            FileNotFoundError: If model file is not found
            Exception: If prediction fails
        """
        if self.model is None:
            logger.error("Model is not loaded. Prediction cannot be performed.")
            raise ValueError(
                "Model is not loaded. Ensure the service is initialized correctly."
            )
        try:
            # Add batch dimension if not present
            predictions = self.model.predict(image)
            logger.debug(f"Raw Predictions: {predictions}")
            confidence = np.max(predictions, axis=1)[0]  # Get max probability
            prediction_index = predictions.argmax(axis=1)[0]

            return CLASS_LABELS[prediction_index], confidence

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise Exception(f"Prediction failed: {str(e)}")

    def get_model_info(self):
        """
        Returns information about the loaded model, such as its type or metadata.

        Returns:
            dict: A dictionary containing model information.
        """

        if self.model is None:
            logger.warning("Model info requested but the model is not loaded.")
            return {"status": "Model not loaded"}
        model_info = {"status": "Model loaded", "model_type": type(self.model).__name__}
        logger.info(f"Model Info: {model_info}")
        return model_info


# # for test
# # python -m app.src.classifier
# if __name__ == "__main__":
#     from tensorflow.keras.preprocessing import image_dataset_from_directory
#     import numpy as np
#     import os
#     from PIL import Image

#     try:
#         # Use os.path.join for cross-platform compatibility
#         base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
#         img_path = os.path.join(
#             base_dir, "test", "1. Enfeksiyonel", "03ContactDerm040127.jpg"
#         )

#         # Check if image exists
#         if not os.path.exists(img_path):
#             raise FileNotFoundError(f"Image not found at {img_path}")

#         processed_image = preprocess_image(img_path)

#         # Save preprocessed image for verification
#         debug_dir = os.path.join(base_dir, "debug")
#         os.makedirs(debug_dir, exist_ok=True)

#         # Convert preprocessed image back to PIL format and save
#         debug_image = Image.fromarray((processed_image[0] * 255).astype("uint8"))
#         debug_path = os.path.join(debug_dir, "preprocessed_image.png")
#         debug_image.save(debug_path)
#         print(f"Saved preprocessed image to: {debug_path}")

#         # Initialize the classifier and make prediction
#         classifier = Classifier()
#         label, confidence = classifier.predict(processed_image)
#         print(f"Predicted disease: {label}")
#         print(f"Confidence: {confidence:.2%}")

#     except Exception as e:
#         print(f"Error: {str(e)}")
