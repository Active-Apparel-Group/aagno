# Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore


# Set working directory inside container
WORKDIR /app

# Copy only the minimal files needed to install
COPY requirements.lock.txt requirements.lock.txt

# Install uv and dependencies
RUN pip install uv && \
    uv pip install --system -r requirements.lock.txt

# Copy actual app code
COPY . .

# Default command
CMD ["streamlit", "run", "cookbook/examples/apps/agentic_rag/app.py", "--server.port=8501", "--server.address=0.0.0.0"]