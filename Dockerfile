# Stage 1: Build the Web Interface
FROM node:20-slim AS web-build
WORKDIR /web
COPY ink-alchemist-web/package*.json ./
RUN npm install
COPY ink-alchemist-web/ ./
RUN npm run build

# Stage 2: Final Image with Python and PyTorch
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy AI pipeline requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy built web assets from Stage 1
COPY --from=web-build /web/dist ./ink-alchemist-web/dist

# Copy AI pipeline code and server script
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
EXPOSE 8080

# Default command: Serve the application
CMD ["python", "serve_app.py"]
