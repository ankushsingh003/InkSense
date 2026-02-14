
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('caption', views.caption_image, name='caption'),
    path('health', views.health, name='health'),
]
