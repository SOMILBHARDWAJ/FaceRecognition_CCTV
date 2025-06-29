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

        status, name_or_id = self.recognition.recognize_face(frame)
        if status == 'no_persons_found':
            util.msg_box("Error", "No face detected. Please try again.")
        elif status == 'multiple_faces_detected':
            util.msg_box("Error", "Multiple faces detected. Ensure only one person is in front of the camera.")
        elif status == 'unknown_person':
            util.msg_box("Error", "Face not recognized. Please register first.")
        else:
            name = status
            emp_id = name_or_id
            util.msg_box('Welcome back!', f'Welcome, {name} (ID: {emp_id}).')
            with open(self.log_path, 'a') as f:
                f.write(f'{name},{emp_id},{datetime.datetime.now()},in\n')
            self.app.current_user = name
            self.app.logged_in_emp_ids.add(emp_id)
            self.app.timer_manager.start()

    def login_threaded(self):
        threading.Thread(target=self.login).start()