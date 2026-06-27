# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies required for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories with proper permissions
RUN mkdir -p voice_assistant/temp_audio voice_assistant/dashboard_state voice_assistant/recordings voice_assistant/plots

# Hugging Face Spaces requires apps to run on port 7860 by default
ENV PORT=7860

# Run the FastAPI server
# (The server code will automatically bind to 0.0.0.0 and use the PORT env var)
CMD ["python", "voice_assistant/server.py"]
