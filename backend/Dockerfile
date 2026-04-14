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

# Copy requirements first to leverage Docker cache
COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt

# Copy the rest of the code
COPY --chown=user . .

# Ensure data directories exist
RUN mkdir -p data/models data/faiss_index

# Add local bin to path
ENV PATH="/home/user/.local/bin:${PATH}"

# Expose Hugging Face's default port
EXPOSE 7860

# Startup command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
