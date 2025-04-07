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
CLASS_LABELS = ["1. Enfeksiyonel", "5. Benign", "6. Malign"]


class Classifier:
    """
    A service class to manage machine learning model loading and predictions.
    Ensures thread-safe operations and provides reusable methods for interacting with the model.
    """

    # Lock for the model in thread to ensure thread safety
    _model_lock = threading.Lock()

    def __init__(self, model_path="model/bestmodel_93.keras"):
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
            print(f"predictions: {predictions}")
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


if __name__ == "__main__":
    from tensorflow.keras.preprocessing import image_dataset_from_directory
    import numpy as np

    try:
        # Update image path to be relative to script location
        img_path = "/root/freelance/skin_diseases_classification/test/6. Malign/ISIC_0056124.jpg"
        processed_image = preprocess_image(img_path)

        # Initialize the classifier
        classifier = Classifier()  # Using the default model path
        test_dir = "test"
        test_data = image_dataset_from_directory(
            test_dir,
            label_mode="categorical",
            image_size=(400, 400),
            batch_size=64,
            shuffle=True,
            seed=42,
        )
        loss, acc = classifier.model.evaluate(test_data)

        print(f"\nAccuracy = {acc}\nLoss = {loss}")
        # # Make prediction
        # label, index = classifier.predict(processed_image)
        # print(f"Predicted disease: {label} (class {index})")

    except Exception as e:
        print(f"Error: {str(e)}")
