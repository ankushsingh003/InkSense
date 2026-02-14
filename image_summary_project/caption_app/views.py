
import os
import json
import requests
from io import BytesIO
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from PIL import Image

# Import ML model
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.captioning_model import get_caption

def index(request):
    """Render main page"""
    return render(request, 'index.html')

@csrf_exempt
@require_http_methods(["POST"])
def caption_image(request):
    """Generate caption for uploaded image or URL"""
    try:
        image = None
        
        # Handle JSON (URL)
        if request.content_type == 'application/json':
            data = json.loads(request.body)
            image_url = data.get('image_url')
            if not image_url:
                return JsonResponse({'success': False, 'error': 'No URL provided'})
            
            try:
                response = requests.get(image_url, timeout=10)
                image = Image.open(BytesIO(response.content))
            except Exception as e:
                return JsonResponse({'success': False, 'error': f'Failed to load URL: {str(e)}'})
                
        # Handle File Upload
        elif 'file' in request.FILES:
            file = request.FILES['file']
            image = Image.open(file)
            
        else:
            return JsonResponse({'success': False, 'error': 'No image provided'})
            
        # Generate Caption
        caption = get_caption(image)
        
        return JsonResponse({
            'success': True,
            'caption': caption
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

def health(request):
    return JsonResponse({'status': 'ok'})
