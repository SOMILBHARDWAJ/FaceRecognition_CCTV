import util
import time
import math


class RecognitionHandler:
    def __init__(self, db_dir, known_encodings=None, known_names=None, multi_encodings_dict=None):
        self.db_dir = db_dir
        self.known_encodings = known_encodings or []
        self.known_names = known_names or []
        self.multi_encodings_dict = multi_encodings_dict or {}

        # Anti-spoofing tracking
        self.bbox_history = []
        self.static_start_time = None
        self.static_threshold = 5.0  # 5 seconds of static detection
        self.movement_threshold = 5  # pixels
        self.max_history = 10  # Keep last 10 bounding boxes

    def reload_known_faces(self):
        self.known_encodings, self.known_names, self.multi_encodings_dict = util.load_known_faces(self.db_dir)
        print(f"Reloaded faces: {len(self.known_names)} users")
        print(f"User names: {self.known_names}")

    def recognize_face(self, frame, use_multi_encodings=False):
        # Standard recognition without bbox (for login/logout)
        return util.recognize(
            frame,
            self.db_dir,
            self.known_encodings,
            self.known_names,
            use_multi_encodings=use_multi_encodings,
            return_bbox=False
        )

    def recognize_face_with_bbox(self, frame, use_multi_encodings=False):
        # Enhanced recognition with bbox for anti-spoofing
        return util.recognize(
            frame,
            self.db_dir,
            self.known_encodings,
            self.known_names,
            use_multi_encodings=use_multi_encodings,
            return_bbox=True
        )

    def calculate_bbox_center(self, bbox):
        """Calculate center point of bounding box"""
        if bbox is None:
            return None
        top, right, bottom, left = bbox
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        return (center_x, center_y)

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        if point1 is None or point2 is None:
            return float('inf')
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def is_bbox_static(self, current_bbox):
        """
        Check if bounding box has been static for too long
        Returns: (is_static, is_spoofing_detected)
        """
        if current_bbox is None:
            self.bbox_history.clear()
            self.static_start_time = None
            return False, False

        current_center = self.calculate_bbox_center(current_bbox)
        current_time = time.time()

        # Add current bbox to history
        self.bbox_history.append({
            'bbox': current_bbox,
            'center': current_center,
            'timestamp': current_time
        })

        # Keep only recent history
        if len(self.bbox_history) > self.max_history:
            self.bbox_history.pop(0)

        # Need at least 3 points to determine movement
        if len(self.bbox_history) < 3:
            return False, False

        # Check if recent positions are static
        recent_centers = [entry['center'] for entry in self.bbox_history[-5:]]
        is_currently_static = True

        # Compare recent centers with current center
        for center in recent_centers[:-1]:  # Exclude current center
            distance = self.calculate_distance(current_center, center)
            if distance > self.movement_threshold:
                is_currently_static = False
                break

        # Handle static detection timing
        if is_currently_static:
            if self.static_start_time is None:
                self.static_start_time = current_time

            static_duration = current_time - self.static_start_time

            # Check if static for too long (spoofing detected)
            if static_duration >= self.static_threshold:
                return True, True  # Static and spoofing detected
            else:
                return True, False  # Static but within allowed time
        else:
            # Movement detected, reset static timer
            self.static_start_time = None
            return False, False

    def reset_anti_spoofing(self):
        """Reset anti-spoofing tracking (call on login/logout)"""
        self.bbox_history.clear()
        self.static_start_time = None

    def get_static_duration(self):
        """Get how long the face has been static"""
        if self.static_start_time is None:
            return 0
        return time.time() - self.static_start_time