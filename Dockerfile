# Stage 1: build the static web demo (client-side heuristic preview)
FROM node:20-slim AS web-build
WORKDIR /web
COPY ink-alchemist-web/package*.json ./
RUN npm install
COPY ink-alchemist-web/ ./
RUN npm run build

# Stage 2: serve the static files. serve_app.py only uses the Python standard library,
# so no ML dependencies are installed here. The trained model is NOT served by this image;
# training and evaluation run through train.py / evaluation.py (see README).
FROM python:3.11-slim
WORKDIR /app
COPY serve_app.py .
COPY --from=web-build /web/dist ./ink-alchemist-web/dist
ENV PYTHONUNBUFFERED=1
ENV PORT=3000
CMD ["python", "serve_app.py"]
