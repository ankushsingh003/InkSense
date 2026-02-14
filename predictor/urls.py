"""
URL configuration for predictor app
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict', views.predict, name='predict'),
    path('health', views.health, name='health'),
]
