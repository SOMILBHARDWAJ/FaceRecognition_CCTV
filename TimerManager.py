import util
from timing_counters import update_attendance, get_user_timer_data
import json
import threading

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
        self._schedule_update()

    def stop(self):
        if self.job_id:
            self.app.main_window.after_cancel(self.job_id)
            self.job_id = None
        # Reset UI labels handled by App

    def _schedule_update(self):
        self._perform_update()

    def _perform_update(self):
        if not self.app.current_user:
            return
        def recognition_task():
            frame = self.app.webcam.get_latest_frame()
            status, emp_id_detected = self.recognition.recognize_face(frame, use_multi_encodings=True)
            is_present = (status == self.app.current_user)
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
                    self.app.label_name = tk.Label(self.app.main_window, text=f"Name: {self.app.current_user}", font=("Helvetica", 12))
                    self.app.label_name.place(x=750, y=120)
                else:
                    self.app.label_name.config(text=f"Name: {self.app.current_user}")
                if not hasattr(self.app, 'label_emp_id'):
                    self.app.label_emp_id = tk.Label(self.app.main_window, text=f"Emp ID: {emp_id}", font=("Helvetica", 12))
                    self.app.label_emp_id.place(x=750, y=150)
                else:
                    self.app.label_emp_id.config(text=f"Emp ID: {emp_id}")
                if missed > 0 and missed > self.alert_threshold and missed % 30 == 0:
                    util.msg_box("Warning!", f"{self.app.current_user} has been absent for {missed} seconds!")
                    self.alert_threshold = missed
            self.app.main_window.after(0, update_ui)
            # Schedule next
            self.job_id = self.app.main_window.after(self.interval_ms, self._perform_update)
        threading.Thread(target=recognition_task).start()
