@echo off
REM Django Development Server Startup Script

echo ======================================
echo Django App - Image Value Prediction
echo ======================================

REM Run migrations (create database tables)
echo.
echo Running migrations...
python manage.py makemigrations
python manage.py migrate

REM Create media directory if it doesn't exist
if not exist "media" mkdir media

echo.
echo ======================================
echo Starting Django Development Server
echo ======================================
echo.
echo Server will be available at:
echo   http://localhost:8000
echo.
echo Press CTRL+C to stop the server
echo ======================================
echo.

REM Start the development server
python manage.py runserver 0.0.0.0:8001
