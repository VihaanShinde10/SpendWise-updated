FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and switch to non-root user
RUN useradd -m -u 1000 user
USER user
WORKDIR $HOME/app

# Copy requirements first
COPY --chown=user backend/requirements.txt .
RUN pip install --user -r requirements.txt

# Copy only the backend folder contents to the container root
COPY --chown=user backend/ .

# Ensure data directories exist
RUN mkdir -p data/models data/faiss_index

# Add local bin to path
ENV PATH="/home/user/.local/bin:${PATH}"

# Expose Hugging Face's default port
EXPOSE 7860

# Startup command (app.main:app works because we copied the CONTENTS of backend/)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
