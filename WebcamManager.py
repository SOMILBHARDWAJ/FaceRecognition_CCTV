import cv2
from PIL import Image, ImageTk
import numpy as np


class WebcamManager:
    def __init__(self, camera_index=1, update_interval=20):
        self.camera_index = camera_index
        self.update_interval = update_interval
        self.cap = None
        self.frame = None
        self.display_frame = None  # NEW: Frame with bounding box drawn
        self.running = False
        self.label = None

        # NEW: Bounding box and status tracking
        self.current_bbox = None
        self.bbox_color = (0, 255, 0)  # Green by default
        self.status_text = ""
        self.show_bbox = False  # Only show after login

    def start(self, label):
        self.label = label
        self.cap = cv2.VideoCapture(self.camera_index)
        self.running = True
        self._update_frame()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def _update_frame(self):
        if not self.running or not self.cap:
            return
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame

            # Create display frame with bounding box if needed
            self.display_frame = frame.copy()

            if self.show_bbox and self.current_bbox is not None:
                self._draw_bounding_box()

            # Convert and display
            img_rgb = cv2.cvtColor(self.display_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            imgtk = ImageTk.PhotoImage(image=pil_img)

            # Avoid garbage collection
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        # Schedule next update
        self.label.after(self.update_interval, self._update_frame)

    def _draw_bounding_box(self):
        """Draw bounding box and status on the display frame"""
        if self.current_bbox is None:
            return

        top, right, bottom, left = self.current_bbox

        # Draw bounding box rectangle
        cv2.rectangle(self.display_frame, (left, top), (right, bottom), self.bbox_color, 2)

        # Draw corner markers for better visibility
        corner_length = 20
        corner_thickness = 3

        # Top-left corner
        cv2.line(self.display_frame, (left, top), (left + corner_length, top), self.bbox_color, corner_thickness)
        cv2.line(self.display_frame, (left, top), (left, top + corner_length), self.bbox_color, corner_thickness)

        # Top-right corner
        cv2.line(self.display_frame, (right, top), (right - corner_length, top), self.bbox_color, corner_thickness)
        cv2.line(self.display_frame, (right, top), (right, top + corner_length), self.bbox_color, corner_thickness)

        # Bottom-left corner
        cv2.line(self.display_frame, (left, bottom), (left + corner_length, bottom), self.bbox_color, corner_thickness)
        cv2.line(self.display_frame, (left, bottom), (left, bottom - corner_length), self.bbox_color, corner_thickness)

        # Bottom-right corner
        cv2.line(self.display_frame, (right, bottom), (right - corner_length, bottom), self.bbox_color,
                 corner_thickness)
        cv2.line(self.display_frame, (right, bottom), (right, bottom - corner_length), self.bbox_color,
                 corner_thickness)

        # Add status text above the bounding box
        if self.status_text:
            # Calculate text position
            text_y = max(top - 10, 20)  # Position above box, but not off-screen

            # Get text size for background rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(self.status_text, font, font_scale, thickness)

            # Draw background rectangle for text
            cv2.rectangle(self.display_frame,
                          (left, text_y - text_height - 5),
                          (left + text_width + 10, text_y + 5),
                          (0, 0, 0), -1)  # Black background

            # Draw text
            cv2.putText(self.display_frame, self.status_text,
                        (left + 5, text_y), font, font_scale, (255, 255, 255), thickness)

    def update_bbox_info(self, bbox, color, status_text):
        """Update bounding box information"""
        self.current_bbox = bbox
        self.bbox_color = color
        self.status_text = status_text

    def enable_bbox_display(self):
        """Enable bounding box display (called after login)"""
        self.show_bbox = True

    def disable_bbox_display(self):
        """Disable bounding box display (called after logout)"""
        self.show_bbox = False
        self.current_bbox = None
        self.status_text = ""

    def get_latest_frame(self):
        return self.frame