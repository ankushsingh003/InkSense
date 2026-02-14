
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image

class ImageCaptioner:
    def __init__(self, model_name="nlpconnect/vit-gpt2-image-captioning", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self.feature_extractor = None
        self.tokenizer = None
        
        print(f"initialized ImageCaptioner on {self.device}")

    def load_model(self):
        """Lazy load the model"""
        if self.model is None:
            print(f"Loading model: {self.model_name}...")
            try:
                self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device)
                self.feature_extractor = ViTImageProcessor.from_pretrained(self.model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {e}")
                raise e

    def predict(self, image_path_or_obj):
        """
        Generate caption for an image
        args:
            image_path_or_obj: str (path) or PIL.Image object
        """
        self.load_model()
        
        try:
            if isinstance(image_path_or_obj, str):
                image = Image.open(image_path_or_obj)
            else:
                image = image_path_or_obj
            
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Preprocess
            pixel_values = self.feature_extractor(images=[image], return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate
            output_ids = self.model.generate(pixel_values, max_length=16, num_beams=4)
            preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            return preds[0].strip()
            
        except Exception as e:
            print(f"Error checking prediction: {e}")
            return f"Error: {str(e)}"

# Singleton instance
captioner = ImageCaptioner()

def get_caption(image):
    return captioner.predict(image)
