
@echo off
echo Starting Image Summary Web App...
cd /d "%~dp0\image_summary_project"

REM Start the development server on port 8002
python manage.py runserver 0.0.0.0:8002
pause
