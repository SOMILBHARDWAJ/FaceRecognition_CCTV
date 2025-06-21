import util
from timing_counters import update_attendance, get_user_timer_data
import json
import threading
import cv2
import tkinter as tk


class TimerManager:
    def __init__(self, app, recognition_handler, users_file_path):
        self.app = app
        self.recognition = recognition_handler
        self.users_file_path = users_file_path
        self.job_id = None
        self.alert_threshold = 0
        self.interval_ms = 5000

    def start(self):
        self.alert_threshold = 0
        # Enable bounding box display when timer starts
        self.app.webcam.enable_bbox_display()
        self._schedule_update()

    def stop(self):
        if self.job_id:
            self.app.main_window.after_cancel(self.job_id)
            self.job_id = None
        # Disable bounding box display when timer stops
        self.app.webcam.disable_bbox_display()

    def _schedule_update(self):
        self._perform_update()

    def _perform_update(self):
        if not self.app.current_user:
            return

        def recognition_task():
            frame = self.app.webcam.get_latest_frame()

            # Use enhanced recognition with bbox for anti-spoofing
            result = self.recognition.recognize_face_with_bbox(frame, use_multi_encodings=True)

            if len(result) == 3:
                status, emp_id_detected, bbox = result
            else:
                # Fallback for compatibility
                status, emp_id_detected = result
                bbox = None

            # Check if face is recognized as current user
            is_face_recognized = (status == self.app.current_user)

            # Anti-spoofing check
            is_spoofing = False
            static_duration = 0

            if is_face_recognized and bbox is not None:
                is_static, is_spoofing = self.recognition.is_bbox_static(bbox)
                static_duration = self.recognition.get_static_duration()

                if is_spoofing:
                    # Spoofing detected - treat as absent
                    is_face_recognized = False
                    print(f"Anti-spoofing: Static face detected for {self.app.current_user}")

            # Determine bounding box color and status text
            bbox_color = (0, 255, 0)  # Green (BGR format)
            status_text = ""

            if bbox is not None:
                if is_spoofing:
                    bbox_color = (0, 0, 255)  # Red
                    status_text = f"⚠️ STATIC DETECTED ({static_duration:.1f}s)"
                elif is_face_recognized:
                    bbox_color = (0, 255, 0)  # Green
                    status_text = f"✅ {self.app.current_user} - ACTIVE"
                else:
                    bbox_color = (0, 165, 255)  # Orange
                    status_text = "❌ FACE NOT RECOGNIZED"
            else:
                # No face detected
                bbox_color = (128, 128, 128)  # Gray
                status_text = "⚠️ NO FACE DETECTED"

            # Update webcam display with bounding box info
            self.app.webcam.update_bbox_info(bbox, bbox_color, status_text)

            # Final presence determination
            is_present = is_face_recognized and not is_spoofing

            # Update attendance
            update_attendance(self.app.current_user, is_present)
            timers = get_user_timer_data(self.app.current_user)
            present = timers['presentCounter']
            absent = timers['absentCounter']
            missed = timers['absentTimeCounter']

            # Read emp_id
            try:
                with open(self.users_file_path, 'r') as f:
                    users_data = json.load(f)
                    emp_id = users_data.get(self.app.current_user, "N/A")
            except:
                emp_id = "N/A"

            # Update UI
            def update_ui():
                self.app.label_present_time.config(text=f"Present: {present}s")
                self.app.label_absent_time.config(text=f"Absent: {absent}s")
                self.app.label_total_missed.config(text=f"Total Missed: {missed}s")

                if not hasattr(self.app, 'label_name'):
                    self.app.label_name = tk.Label(self.app.main_window, text=f"Name: {self.app.current_user}",
                                                   font=("Helvetica", 12))
                    self.app.label_name.place(x=750, y=120)
                else:
                    self.app.label_name.config(text=f"Name: {self.app.current_user}")

                if not hasattr(self.app, 'label_emp_id'):
                    self.app.label_emp_id = tk.Label(self.app.main_window, text=f"Emp ID: {emp_id}",
                                                     font=("Helvetica", 12))
                    self.app.label_emp_id.place(x=750, y=150)
                else:
                    self.app.label_emp_id.config(text=f"Emp ID: {emp_id}")

                # Enhanced anti-spoofing status indicator
                if not hasattr(self.app, 'label_status'):
                    self.app.label_status = tk.Label(self.app.main_window, text="Status: Active",
                                                     font=("Helvetica", 12))
                    self.app.label_status.place(x=750, y=180)

                # Enhanced status display
                if is_spoofing:
                    self.app.label_status.config(
                        text=f"Status: ⚠️ STATIC FACE ({static_duration:.1f}s)",
                        fg="red"
                    )
                elif is_present:
                    self.app.label_status.config(text="Status: ✅ ACTIVE", fg="green")
                elif bbox is None:
                    self.app.label_status.config(text="Status: ⚠️ NO FACE", fg="orange")
                else:
                    self.app.label_status.config(text="Status: ❌ NOT RECOGNIZED", fg="red")

                # Alert for missed time with enhanced anti-spoofing messages
                if missed > 0 and missed > self.alert_threshold and missed % 30 == 0:
                    if is_spoofing:
                        util.msg_box("⚠️ ANTI-SPOOFING ALERT!",
                                     f"Static face detected for {self.app.current_user}!\n\n"
                                     f"Please move naturally in front of the camera.\n"
                                     f"Missed time: {missed}s\n\n"
                                     f"Static duration: {static_duration:.1f}s")
                    else:
                        util.msg_box("⚠️ ATTENDANCE WARNING!",
                                     f"{self.app.current_user} has been absent for {missed} seconds!\n\n"
                                     f"Please return to your workstation.")
                    self.alert_threshold = missed

            self.app.main_window.after(0, update_ui)

            # Schedule next update
            self.job_id = self.app.main_window.after(self.interval_ms, self._perform_update)

        threading.Thread(target=recognition_task).start()