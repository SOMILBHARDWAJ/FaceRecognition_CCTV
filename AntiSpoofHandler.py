import os
import cv2
import numpy as np
import warnings
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')


class AntiSpoofHandler:
    def __init__(self, model_dir="./resources/anti_spoof_models", device_id=0, threshold=0.3):
        """
        Initialize Anti-Spoofing Handler

        Args:
            model_dir: Path to anti-spoof models directory
            device_id: GPU device ID (0 for CPU)
            threshold: Threshold for real face detection (increased to 0.7 --> 0.5 for better security)
        """
        self.model_dir = model_dir
        self.device_id = device_id
        self.threshold = threshold
        self.debug_mode = True  # Enable debug logging

        # Initialize components
        self.model_test = None
        self.image_cropper = None
        self.models_loaded = False

        # Try to initialize models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize anti-spoofing models"""
        try:
            print(f"Checking model directory: {self.model_dir}")

            if not os.path.exists(self.model_dir):
                print(f"ERROR: Model directory does not exist: {self.model_dir}")
                return

            model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pth')]
            print(f"Found model files: {model_files}")

            if not model_files:
                print("ERROR: No .pth model files found in directory")
                return

            self.model_test = AntiSpoofPredict(self.device_id)
            self.image_cropper = CropImage()
            self.models_loaded = True
            print(f"✓ Anti-spoofing models loaded successfully from {self.model_dir}")
            print(f"✓ Using threshold: {self.threshold}")

        except Exception as e:
            print(f"ERROR initializing anti-spoof models: {e}")
            import traceback
            traceback.print_exc()

    def is_real_face(self, image):
        """
        Check if the face in the image is real (not spoofed)

        Args:
            image: OpenCV image (BGR format)

        Returns:
            tuple: (is_real: bool, confidence: float, error_msg: str or None)
        """
        if not self.models_loaded:
            print("WARNING: Anti-spoofing models not loaded - treating as FAKE for security")
            return False, 0.0, "Anti-spoofing disabled - assuming fake"

        try:
            # Validate image
            if image is None or image.size == 0:
                if self.debug_mode:
                    print("DEBUG: Invalid image provided")
                return False, 0.0, "Invalid image"

            # Get face bounding box
            image_bbox = self.model_test.get_bbox(image)
            if image_bbox is None:
                if self.debug_mode:
                    print("DEBUG: No face detected for anti-spoofing")
                return False, 0.0, "No face detected for anti-spoofing"

            if self.debug_mode:
                print(f"DEBUG: Face bbox detected: {image_bbox}")

            # Initialize prediction
            prediction = np.zeros((1, 3))
            model_count = 0

            # Process each model in the directory
            model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pth')]

            for model_name in model_files:
                try:
                    if self.debug_mode:
                        print(f"DEBUG: Processing model: {model_name}")

                    h_input, w_input, model_type, scale = parse_model_name(model_name)

                    param = {
                        "org_img": image,
                        "bbox": image_bbox,
                        "scale": scale,
                        "out_w": w_input,
                        "out_h": h_input,
                        "crop": True,
                    }

                    if scale is None:
                        param["crop"] = False

                    # Crop image for model input
                    img = self.image_cropper.crop(**param)

                    # Get prediction from model
                    model_prediction = self.model_test.predict(img, os.path.join(self.model_dir, model_name))
                    prediction += model_prediction
                    model_count += 1

                    if self.debug_mode:
                        print(f"DEBUG: Model {model_name} prediction: {model_prediction}")

                except Exception as e:
                    print(f"ERROR processing model {model_name}: {e}")
                    continue

            if model_count == 0:
                print("ERROR: No models processed successfully")
                return False, 0.0, "No models processed"

            # Analyze results
            label = np.argmax(prediction)
            confidence = prediction[0][label] / model_count  # Average confidence

            if self.debug_mode:
                print(f"DEBUG: Final prediction: {prediction}")
                print(f"DEBUG: Label: {label} (0=Fake, 1=Real)")
                print(f"DEBUG: Confidence: {confidence:.4f}")
                print(f"DEBUG: Threshold: {self.threshold}")

            # Label 1 = Real Face, Label 0 = Fake Face
            is_real = (label == 1 and confidence >= self.threshold)

            if self.debug_mode:
                print(f"DEBUG: Result - Real: {is_real}")

            return is_real, float(confidence), None

        except Exception as e:
            print(f"ERROR in anti-spoofing: {e}")
            import traceback
            traceback.print_exc()
            # Return False for security - if there's an error, assume it's fake
            return False, 0.0, f"Error: {str(e)}"

    def check_frame_authenticity(self, frame):
        """
        Convenience method to check frame authenticity

        Args:
            frame: OpenCV frame

        Returns:
            dict: {
                'is_authentic': bool,
                'confidence': float,
                'status': str,
                'error': str or None
            }
        """
        is_real, confidence, error = self.is_real_face(frame)

        if error and "disabled" not in error.lower():
            status = "error"
        elif is_real:
            status = "authentic"
        else:
            status = "spoofed"

        if self.debug_mode:
            print(f"DEBUG: Frame authenticity check - Status: {status}, Confidence: {confidence:.4f}")

        return {
            'is_authentic': is_real,
            'confidence': confidence,
            'status': status,
            'error': error
        }

    def test_with_sample_image(self, image_path):
        """
        Test anti-spoofing with a sample image for debugging
        """
        if not os.path.exists(image_path):
            print(f"Test image not found: {image_path}")
            return

        print(f"Testing anti-spoofing with image: {image_path}")
        image = cv2.imread(image_path)
        result = self.check_frame_authenticity(image)
        print(f"Test result: {result}")
        return result

    def enable_debug(self, enable=True):
        """Enable or disable debug mode"""
        self.debug_mode = enable
        print(f"Debug mode {'enabled' if enable else 'disabled'}")

    def get_model_info(self):
        """Get information about loaded models"""
        if not self.models_loaded:
            return "Models not loaded"

        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pth')]
        return {
            'models_loaded': self.models_loaded,
            'model_count': len(model_files),
            'model_files': model_files,
            'threshold': self.threshold,
            'model_dir': self.model_dir
        }