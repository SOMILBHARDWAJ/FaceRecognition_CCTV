# # -*- coding: utf-8 -*-
# # @Time : 20-6-9 下午3:06
# # @Author : zhuying
# # @Company : Minivision
# # @File : test.py
# # @Software : PyCharm
#
# import os
# import cv2
# import numpy as np
# import argparse
# import warnings
# import time
#
# from src.anti_spoof_predict import AntiSpoofPredict
# from src.generate_patches import CropImage
# from src.utility import parse_model_name
# warnings.filterwarnings('ignore')
#
#
# SAMPLE_IMAGE_PATH = "./images/sample/"
#
#
# # 因为安卓端APK获取的视频流宽高比为3:4,为了与之一致，所以将宽高比限制为3:4
# def check_image(image):
#     height, width, channel = image.shape
#     if width/height != 3/4:
#         print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
#         return False
#     else:
#         return True
#
#
# def test(image_name, model_dir, device_id):
#     model_test = AntiSpoofPredict(device_id)
#     image_cropper = CropImage()
#     image = cv2.imread(SAMPLE_IMAGE_PATH + image_name)
#     result = check_image(image)
#     if result is False:
#         return
#     image_bbox = model_test.get_bbox(image)
#     prediction = np.zeros((1, 3))
#     test_speed = 0
#     # sum the prediction from single model's result
#     for model_name in os.listdir(model_dir):
#         h_input, w_input, model_type, scale = parse_model_name(model_name)
#         param = {
#             "org_img": image,
#             "bbox": image_bbox,
#             "scale": scale,
#             "out_w": w_input,
#             "out_h": h_input,
#             "crop": True,
#         }
#         if scale is None:
#             param["crop"] = False
#         img = image_cropper.crop(**param)
#         start = time.time()
#         prediction += model_test.predict(img, os.path.join(model_dir, model_name))
#         test_speed += time.time()-start
#
#     # draw result of prediction
#     label = np.argmax(prediction)
#     value = prediction[0][label]/2
#     if label == 1:
#         print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
#         result_text = "RealFace Score: {:.2f}".format(value)
#         color = (255, 0, 0)
#     else:
#         print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
#         result_text = "FakeFace Score: {:.2f}".format(value)
#         color = (0, 0, 255)
#     print("Prediction cost {:.2f} s".format(test_speed))
#     cv2.rectangle(
#         image,
#         (image_bbox[0], image_bbox[1]),
#         (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
#         color, 2)
#     cv2.putText(
#         image,
#         result_text,
#         (image_bbox[0], image_bbox[1] - 5),
#         cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)
#
#     format_ = os.path.splitext(image_name)[-1]
#     result_image_name = image_name.replace(format_, "_result" + format_)
#     cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image)
#
#
# if __name__ == "__main__":
#     desc = "test"
#     parser = argparse.ArgumentParser(description=desc)
#     parser.add_argument(
#         "--device_id",
#         type=int,
#         default=0,
#         help="which gpu id, [0/1/2/3]")
#     parser.add_argument(
#         "--model_dir",
#         type=str,
#         default="./resources/anti_spoof_models",
#         help="model_lib used to test")
#     parser.add_argument(
#         "--image_name",
#         type=str,
#         default="image_F1.jpg",
#         help="image used to test")
#     args = parser.parse_args()
#     test(args.image_name, args.model_dir, args.device_id)

# !/usr/bin/env python3
"""
Test script for anti-spoofing functionality
"""

import cv2
import os
import sys
from AntiSpoofHandler import AntiSpoofHandler


def test_anti_spoofing():
    """Test anti-spoofing functionality"""

    print("=== Anti-Spoofing Test ===")

    # Initialize anti-spoof handler
    anti_spoof = AntiSpoofHandler()

    # Check if models are loaded
    model_info = anti_spoof.get_model_info()
    print(f"Model Info: {model_info}")

    if not anti_spoof.models_loaded:
        print("❌ ERROR: Anti-spoofing models not loaded!")
        print("Please check:")
        print("1. ./resources/anti_spoof_models directory exists")
        print("2. Model files (.pth) are present in the directory")
        print("3. Required dependencies are installed")
        return False

    print("✅ Anti-spoofing models loaded successfully!")

    # Test with webcam
    print("\n=== Testing with Webcam ===")
    print("Press 'q' to quit, 's' to test current frame")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Could not open webcam")
        return False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Display frame
            cv2.imshow('Anti-Spoofing Test - Press S to test, Q to quit', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                print("\n--- Testing current frame ---")
                result = anti_spoof.check_frame_authenticity(frame)

                print(f"Result: {result}")

                if result['is_authentic']:
                    print("✅ AUTHENTIC: Real face detected")
                    status_text = f"REAL (conf: {result['confidence']:.2f})"
                    color = (0, 255, 0)  # Green
                else:
                    print("🚨 SPOOFED: Fake face detected")
                    status_text = f"FAKE (conf: {result['confidence']:.2f})"
                    color = (0, 0, 255)  # Red

                # Draw result on frame
                cv2.putText(frame, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.imshow('Test Result', frame)
                cv2.waitKey(2000)  # Show result for 2 seconds

            elif key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return True


def test_with_sample_images():
    """Test with sample images if available"""

    print("\n=== Testing with Sample Images ===")

    # Check for sample images
    sample_dir = "./images/sample/"
    if not os.path.exists(sample_dir):
        print(f"Sample directory not found: {sample_dir}")
        return

    image_files = [f for f in os.listdir(sample_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    if not image_files:
        print("No image files found in sample directory")
        return

    anti_spoof = AntiSpoofHandler()

    for image_file in image_files:
        image_path = os.path.join(sample_dir, image_file)
        print(f"\nTesting: {image_file}")

        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            continue

        result = anti_spoof.check_frame_authenticity(image)

        if result['is_authentic']:
            print(f"✅ {image_file}: REAL (confidence: {result['confidence']:.4f})")
        else:
            print(f"🚨 {image_file}: FAKE (confidence: {result['confidence']:.4f})")


def check_dependencies():
    """Check if required dependencies are available"""

    print("=== Checking Dependencies ===")

    try:
        from src.anti_spoof_predict import AntiSpoofPredict
        print("✅ anti_spoof_predict module available")
    except ImportError as e:
        print(f"❌ anti_spoof_predict module missing: {e}")
        return False

    try:
        from src.generate_patches import CropImage
        print("✅ generate_patches module available")
    except ImportError as e:
        print(f"❌ generate_patches module missing: {e}")
        return False

    try:
        from src.utility import parse_model_name
        print("✅ utility module available")
    except ImportError as e:
        print(f"❌ utility module missing: {e}")
        return False

    # Check model directory
    model_dir = "./resources/anti_spoof_models"
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        print(f"✅ Model directory exists with {len(model_files)} model files")
        for model_file in model_files:
            print(f"   - {model_file}")
    else:
        print(f"❌ Model directory not found: {model_dir}")
        return False

    return True


if __name__ == "__main__":
    print("Anti-Spoofing System Test")
    print("=" * 50)

    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Dependency check failed!")
        print("Please ensure all required files and modules are present.")
        sys.exit(1)

    print("\n✅ All dependencies check passed!")

    # Test anti-spoofing
    try:
        if test_anti_spoofing():
            print("\n✅ Anti-spoofing test completed successfully!")
        else:
            print("\n❌ Anti-spoofing test failed!")

        # Test with sample images if available
        test_with_sample_images()

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()

    print("\nTest completed!")