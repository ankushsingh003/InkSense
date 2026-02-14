"""
Tests for predictor app
"""
from django.test import TestCase, Client
from django.urls import reverse


class PredictorTests(TestCase):
    """Test cases for price prediction"""
    
    def setUp(self):
        self.client = Client()
    
    def test_index_page(self):
        """Test that index page loads"""
        response = self.client.get(reverse('index'))
        self.assertEqual(response.status_code, 200)
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get(reverse('health'))
        self.assertEqual(response.status_code, 200)
        self.assertIn('status', response.json())
    
    def test_predict_no_image(self):
        """Test prediction with no image"""
        response = self.client.post(reverse('predict'))
        self.assertEqual(response.status_code, 400)
