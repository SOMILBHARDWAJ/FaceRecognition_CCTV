import util
from timing_counters import update_attendance, get_user_timer_data
import json
import threading
import time
import tkinter as tk


class TimerManager:
    def __init__(self, app, recognition_handler, users_file_path):
        self.app = app
        self.recognition = recognition_handler
        self.users_file_path = users_file_path
        self.job_id = None
        self.alert_threshold = 0
        self.interval_ms = 5000
        self.spoofing_alert_counter = 0
        self.consecutive_spoofing_count = 0
        self.debug_mode = True  # Enable debug logging

    def start(self):
        self.alert_threshold = 0
        self.spoofing_alert_counter = 0
        self.consecutive_spoofing_count = 0
        print("TimerManager started - monitoring for spoofing attempts")
        self._schedule_update()

    def stop(self):
        if self.job_id:
            self.app.main_window.after_cancel(self.job_id)
            self.job_id = None
        print("TimerManager stopped")

    def _schedule_update(self):
        self._perform_update()

    def _perform_update(self):
        if not self.app.current_user:
            return

        def recognition_task():
            try:
                frame = self.app.webcam.get_latest_frame()
                if frame is None:
                    print("Warning: No frame available from webcam")
                    self.job_id = self.app.main_window.after(self.interval_ms, self._perform_update)
                    return

                # First check for face recognition
                status, emp_id_detected = self.recognition.recognize_face(frame, use_multi_encodings=True)
                face_recognized = (status == self.app.current_user)

                if self.debug_mode:
                    print(
                        f"Face recognition status: {status}, Expected: {self.app.current_user}, Match: {face_recognized}")

                # Initialize presence status
                is_present = False
                spoof_detected = False

                # If face is recognized, check for anti-spoofing
                if face_recognized:
                    if self.debug_mode:
                        print("Face recognized - checking for spoofing...")

                    # Check if face is authentic (not spoofed)
                    spoof_result = self.app.anti_spoof_handler.check_frame_authenticity(frame)

                    if self.debug_mode:
                        print(f"Anti-spoof result: {spoof_result}")

                    if spoof_result['is_authentic']:
                        is_present = True
                        self.consecutive_spoofing_count = 0  # Reset spoofing counter
                        if self.debug_mode:
                            print("✓ Face is authentic - marking as present")
                    else:
                        # Face recognized but spoofed - mark as absent
                        spoof_detected = True
                        self.consecutive_spoofing_count += 1
                        print(f"🚨 SPOOFING DETECTED for {self.app.current_user}: {spoof_result['status']} "
                              f"(confidence: {spoof_result['confidence']:.2f}) - Count: {self.consecutive_spoofing_count}")

                        # Log spoofing attempt
                        self._log_spoofing_attempt(spoof_result)

                else:
                    # Face not recognized at all
                    if self.debug_mode:
                        print("Face not recognized - marking as absent")

                # Update attendance based on presence status
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

                    # Update name and emp_id labels
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

                    # Add security status label
                    if not hasattr(self.app, 'label_security_status'):
                        self.app.label_security_status = tk.Label(self.app.main_window, text="Security: OK",
                                                                  font=("Helvetica", 10), fg="green")
                        self.app.label_security_status.place(x=750, y=180)

                    # Update security status
                    if spoof_detected:
                        self.app.label_security_status.config(text="Security: SPOOFING DETECTED", fg="red")
                    elif face_recognized and is_present:
                        self.app.label_security_status.config(text="Security: AUTHENTICATED", fg="green")
                    elif face_recognized and not is_present:
                        self.app.label_security_status.config(text="Security: FACE NOT DETECTED", fg="orange")
                    else:
                        self.app.label_security_status.config(text="Security: NOT PRESENT", fg="gray")

                    # Absence alert
                    if missed > 0 and missed > self.alert_threshold and missed % 30 == 0:
                        util.msg_box("Warning!", f"{self.app.current_user} has been absent for {missed} seconds!")
                        self.alert_threshold = missed

                    # Spoofing detection alerts
                    if spoof_detected:
                        self.spoofing_alert_counter += 1

                        # Immediate alert for first spoofing detection
                        if self.spoofing_alert_counter == 1:
                            util.msg_box("🚨 SECURITY ALERT!",
                                         f"Spoofing attempt detected for {self.app.current_user}!\n"
                                         f"Please use live camera, not photos/videos.")

                        # Periodic alerts for continued spoofing
                        elif self.spoofing_alert_counter % 6 == 0:  # Every 30 seconds
                            util.msg_box("🚨 CONTINUED SPOOFING!",
                                         f"Multiple spoofing attempts detected for {self.app.current_user}!\n"
                                         f"Count: {self.consecutive_spoofing_count}\n"
                                         f"Please use live camera only.")

                self.app.main_window.after(0, update_ui)

            except Exception as e:
                print(f"Error in recognition task: {e}")
                import traceback
                traceback.print_exc()

            # Schedule next update
            self.job_id = self.app.main_window.after(self.interval_ms, self._perform_update)

        threading.Thread(target=recognition_task, daemon=True).start()

    def _log_spoofing_attempt(self, spoof_result):
        """Log spoofing attempts to a file"""
        try:
            log_entry = f"{time.strftime('%Y-%m-%d %H:%M:%S')},{self.app.current_user}," \
                        f"SPOOFING_ATTEMPT,{spoof_result['status']},{spoof_result['confidence']:.4f}\n"

            with open("spoofing_log.txt", "a") as f:
                f.write(log_entry)

        except Exception as e:
            print(f"Error logging spoofing attempt: {e}")

    def enable_debug(self, enable=True):
        """Enable or disable debug mode"""
        self.debug_mode = enable
        print(f"TimerManager debug mode {'enabled' if enable else 'disabled'}")

    def get_spoofing_stats(self):
        """Get spoofing detection statistics"""
        return {
            'spoofing_alert_counter': self.spoofing_alert_counter,
            'consecutive_spoofing_count': self.consecutive_spoofing_count,
            'current_user': self.app.current_user
        }