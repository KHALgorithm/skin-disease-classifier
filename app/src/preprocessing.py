import numpy as np
import cv2
from typing import Union
import os
from tensorflow.keras.applications.vgg19 import preprocess_input


def preprocess_image(image: Union[str, np.ndarray]) -> np.ndarray:
    """
    Preprocess image for VGG19 model inference. Handles both file paths and numpy arrays.

    Args:
        image: Either a file path (str) or numpy array of the image

    Returns:
        np.ndarray: Preprocessed image array ready for VGG19 model inference

    Raises:
        ValueError: If image format is invalid or processing fails
        FileNotFoundError: If image file doesn't exist
    """
    try:
        target_size = (400, 400)  # VGG19 expected input size

        # Handle file path input
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")

            # Load and convert to RGB
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Failed to load image: {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Handle numpy array input
        elif isinstance(image, np.ndarray):
            img = image.copy()
            # Convert BGR to RGB if needed (check if image has 3 channels)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Input must be either a file path or numpy array")

        # Resize to VGG19 expected size
        img = cv2.resize(img, target_size)

        # Convert to float32
        img = img.astype("float32")

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        # Apply VGG19 specific preprocessing
        img = preprocess_input(img)

        return img

    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")
