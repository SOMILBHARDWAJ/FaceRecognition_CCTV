import util
import datetime
import threading


class LoginHandler:
    def __init__(self, app, recognition_handler, log_path):
        self.app = app
        self.recognition = recognition_handler
        self.log_path = log_path

    def login(self):
        if self.app.current_user:
            util.msg_box("Already Logged In", f"User '{self.app.current_user}' is already logged in.")
            return

        frame = self.app.webcam.get_latest_frame()

        # Enhanced login with anti-spoofing check
        result = self.recognition.recognize_face_with_bbox(frame, use_multi_encodings=False)

        if len(result) == 3:
            status, name_or_id, bbox = result
        else:
            # Fallback for compatibility
            status, name_or_id = result
            bbox = None

        if status == 'no_persons_found':
            util.msg_box("❌ Login Failed",
                         "No face detected. Please position yourself in front of the camera and try again.")
        elif status == 'multiple_faces_detected':
            util.msg_box("❌ Login Failed", "Multiple faces detected. Ensure only one person is in front of the camera.")
        elif status == 'unknown_person':
            util.msg_box("❌ Login Failed", "Face not recognized. Please register first or contact administrator.")
        else:
            name = status
            emp_id = name_or_id

            # Additional anti-spoofing check at login
            if bbox is not None:
                # Check if face seems too static (basic check)
                is_static, is_spoofing = self.recognition.is_bbox_static(bbox)

                if is_spoofing:
                    util.msg_box("⚠️ ANTI-SPOOFING WARNING",
                                 f"Please move naturally in front of the camera.\n\n"
                                 f"Static face detection activated.\n"
                                 f"Try logging in again with natural movement.")
                    return

            util.msg_box('🎉 Login Successful!',
                         f'Welcome back, {name}!\n'
                         f'Employee ID: {emp_id}\n\n'
                         f'Anti-spoofing monitoring is now active.')

            with open(self.log_path, 'a') as f:
                f.write(f'{name},{emp_id},{datetime.datetime.now()},in\n')

            self.app.current_user = name
            self.app.logged_in_emp_ids.add(emp_id)

            # Reset anti-spoofing tracking for new login
            self.recognition.reset_anti_spoofing()

            # Start timer (which will enable bounding box display)
            self.app.timer_manager.start()

    def login_threaded(self):
        threading.Thread(target=self.login).start()