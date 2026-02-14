"""
Django views for price prediction
Converted from Flask app
"""
import os
import json
import torch
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model import load_model
from src.predict import predict_single_image
from src.utils import load_image, format_price

# Global model (loaded on first request)
model = None
device = None
model_loaded = False


def get_model():
    """Lazy load model on first request"""
    global model, device, model_loaded
    
    if model is None:
        device = torch.device(config.DEVICE)
        
        if os.path.exists(config.BEST_MODEL_PATH):
            model = load_model(checkpoint_path=config.BEST_MODEL_PATH, device=device)
            model_loaded = True
            print("Model loaded successfully!")
        else:
            model = load_model(device=device)
            model_loaded = False
            print("Model not trained - using random weights")
    
    return model, device, model_loaded


def index(request):
    """Render main page"""
    # Check if model exists without loading it
    model_exists = os.path.exists(config.BEST_MODEL_PATH)
    context = {'model_loaded': model_exists}
    return render(request, 'index.html', context)


@csrf_exempt
@require_http_methods(["POST"])
def predict(request):
    """Prediction API endpoint"""
    try:
        # Load model on first request
        model, device, model_loaded = get_model()
        
        image_path = None
        temp_file = None
        
        # Check if request contains JSON data (URL)
        if request.content_type == 'application/json':
            try:
                data = json.loads(request.body)
                image_url = data.get('image_url')
                
                if not image_url:
                    return JsonResponse({
                        'success': False,
                        'error': 'No image URL provided'
                    }, status=400)
                
                # Predict directly from URL
                predicted_price = predict_single_image(image_url, model, device)
            
            except json.JSONDecodeError:
                return JsonResponse({
                    'success': False,
                    'error': 'Invalid JSON'
                }, status=400)
        
        # Check if file is uploaded
        elif 'file' in request.FILES:
            file = request.FILES['file']
            
            if not file:
                return JsonResponse({
                    'success': False,
                    'error': 'No file selected'
                }, status=400)
            
            # Check file extension
            allowed_ext = config.ALLOWED_EXTENSIONS
            file_ext = file.name.rsplit('.', 1)[1].lower() if '.' in file.name else ''
            
            if file_ext not in allowed_ext:
                return JsonResponse({
                    'success': False,
                    'error': f'Invalid file type. Allowed: {", ".join(allowed_ext)}'
                }, status=400)
            
            # Save temporary file
            temp_file = f'temp_upload.{file_ext}'
            file_path = default_storage.save(temp_file, ContentFile(file.read()))
            full_path = os.path.join(settings.MEDIA_ROOT, file_path)
            
            # Predict
            predicted_price = predict_single_image(full_path, model, device)
            
            # Clean up
            if os.path.exists(full_path):
                os.remove(full_path)
        
        else:
            return JsonResponse({
                'success': False,
                'error': 'No image provided'
            }, status=400)
        
        # Check if prediction was successful
        if predicted_price is None:
            return JsonResponse({
                'success': False,
                'error': 'Prediction failed'
            }, status=500)
        
        # Return result
        return JsonResponse({
            'success': True,
            'predicted_price': float(predicted_price),
            'formatted_price': format_price(predicted_price),
            'currency': 'USD',
            'model_trained': model_loaded
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def health(request):
    """Health check endpoint"""
    _, device, model_loaded = get_model()
    return JsonResponse({
        'status': 'ok',
        'model_loaded': model_loaded,
        'device': str(device),
        'backbone': config.BACKBONE
    })
