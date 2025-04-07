# """Application configuration management module.

# This module handles all configuration settings for the application using Pydantic BaseSettings.
# Configuration values are loaded from environment variables and/or .env files.
# """

# from dotenv import load_dotenv
# from pydantic_settings import BaseSettings

# from app.core.logger import logger

# # Load environment variables from .env file
# load_dotenv(override=True)
# logger = logger(__name__)


# class Settings(BaseSettings):
#     """Main application settings class."""

#     class Config:
#         """Pydantic configuration class."""

#         env_file = ".env"
#         env_file_encoding = "utf-8"
#         extra = "allow"
#         case_sensitive = False


# # Initialize settings
# settings = Settings()
