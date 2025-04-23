# Use the official Python base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /usr/src/skin_disease_classifier

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt ./

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install the required packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Make port 5353 available to the world outside this container
EXPOSE 5353

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/usr/src/skin_disease_classifier

# Copy application files
COPY app/ ./app/

# Run main.py when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5353"]
