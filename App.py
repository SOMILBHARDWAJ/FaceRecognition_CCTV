import json
import os
import tkinter as tk

import util
from LoginHandler import LoginHandler
from LogoutHandler import LogoutHandler
from RecognitionHandler import RecognitionHandler
from RegistrationHandler import RegistrationHandler
from TimerManager import TimerManager
from WebcamManager import WebcamManager
from AntiSpoofHandler import AntiSpoofHandler

class App:
    def __init__(self):
        self.main_window = tk.Tk()
        screen_width = self.main_window.winfo_screenwidth()
        screen_height = self.main_window.winfo_screenheight()
        window_width, window_height = 1200, 520
        self.x_pos = int((screen_width - window_width) / 2)
        self.y_pos = int((screen_height - window_height) / 2)
        self.main_window.geometry(f"{window_width}x{window_height}+{self.x_pos}+{self.y_pos}")
        self.main_window.title("Face Recognition Attendance System")

        # Initialize DB and logging
        self.db_dir = "face_db"
        os.makedirs(self.db_dir, exist_ok=True)
        self.users_file_path = os.path.join(self.db_dir, 'users.json')
        if not os.path.exists(self.users_file_path) or os.path.getsize(self.users_file_path) == 0:
            with open(self.users_file_path, 'w') as f:
                json.dump({}, f)
        self.log_path = './log.txt'
        self.current_user = None
        self.logged_in_emp_ids = set()

        # Load known faces - FIXED: properly pass all three parameters
        known_encodings, known_names, multi_encodings_dict = util.load_known_faces(self.db_dir)
        self.recognition_handler = RecognitionHandler(
            self.db_dir,
            known_encodings,
            known_names,  # This was missing before
            multi_encodings_dict
        )

        # Replace the threshold to be more strict
        self.anti_spoof_handler = AntiSpoofHandler(threshold=0.7)  # More strict threshold

        # Add this after creating anti_spoof_handler to enable debug mode
        self.anti_spoof_handler.enable_debug(True)

        # Webcam manager
        self.webcam = WebcamManager()

        # UI Buttons
        self.login_handler = LoginHandler(self, self.recognition_handler, self.log_path)
        btn_login = util.get_button(self.main_window, 'Login', 'green', self.login_handler.login_threaded)
        btn_login.place(x=750, y=200)

        self.logout_handler = LogoutHandler(self, self.recognition_handler, self.log_path)
        btn_logout = util.get_button(self.main_window, 'Logout', 'red', self.logout_handler.logout_threaded)
        btn_logout.place(x=750, y=300)

        self.registration_handler = RegistrationHandler(self, self.recognition_handler)
        btn_register = util.get_button(self.main_window, 'Register New User', 'gray',
                                       self.registration_handler.open_window, fg='black')
        btn_register.place(x=750, y=400)

        # Labels for webcam and timers
        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)
        self.webcam.start(self.webcam_label)

        self.label_present_time = tk.Label(self.main_window, text="Present: 0s", font=("Helvetica", 12))
        self.label_present_time.place(x=750, y=30)
        self.label_absent_time = tk.Label(self.main_window, text="Absent: 0s", font=("Helvetica", 12))
        self.label_absent_time.place(x=750, y=60)
        self.label_total_missed = tk.Label(self.main_window, text="Total Missed: 0s", font=("Helvetica", 12))
        self.label_total_missed.place(x=750, y=90)

        # Timer manager
        self.timer_manager = TimerManager(self, self.recognition_handler, self.users_file_path)

        # Window close
        self.main_window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def reset_ui_after_logout(self):
        self.label_present_time.config(text="Present: 0s")
        self.label_absent_time.config(text="Absent: 0s")
        self.label_total_missed.config(text="Total Missed: 0s")
        if hasattr(self, 'label_name'):
            self.label_name.destroy()
            del self.label_name
        if hasattr(self, 'label_emp_id'):
            self.label_emp_id.destroy()
            del self.label_emp_id

    def on_closing(self):
        self.timer_manager.stop()
        self.webcam.stop()
        self.main_window.destroy()

    def start(self):
        self.main_window.mainloop()