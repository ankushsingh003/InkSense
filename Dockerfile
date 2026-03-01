# Stage 1: Build the Web Interface
FROM node:20-slim AS web-build
WORKDIR /web
COPY ink-alchemist-web/package*.json ./
RUN npm install
COPY ink-alchemist-web/ ./
RUN npm run build

# Stage 2: Final Image
# Using a standard python image as the AI logic in serve_app.py is simulated
# and we want to keep the image size small for Render.
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies for OpenCV (if needed for any future backend logic)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy built web assets from Stage 1
COPY --from=web-build /web/dist ./ink-alchemist-web/dist

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
# Render provides the PORT env var, but we'll set a default just in case
ENV PORT=3000

# Default command: Serve the application
CMD ["python", "serve_app.py"]
