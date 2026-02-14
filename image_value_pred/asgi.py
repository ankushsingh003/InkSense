"""
ASGI config for image_value_pred project.
"""
import os
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'image_value_pred.settings')
application = get_asgi_application()
