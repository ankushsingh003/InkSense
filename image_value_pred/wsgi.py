"""
WSGI config for image_value_pred project.
"""
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'image_value_pred.settings')
application = get_wsgi_application()
