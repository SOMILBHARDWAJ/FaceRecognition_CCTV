import util
import datetime
import threading

class LogoutHandler:
    def __init__(self, app, recognition_handler, log_path):
        self.app = app
        self.recognition = recognition_handler
        self.log_path = log_path

    def logout(self):
        if not self.app.current_user:
            util.msg_box("Error", "No user is currently logged in.")
            return
        frame = self.app.webcam.get_latest_frame()
        status, name_or_id = self.recognition.recognize_face(frame)
        if status in ['no_persons_found', 'multiple_faces_detected', 'unknown_person']:
            msg = {
                'no_persons_found': "No face detected. Please try again.",
                'multiple_faces_detected': "Multiple faces detected. Ensure only one person is in front of the camera.",
                'unknown_person': "Face not recognized. Please try again."
            }
            util.msg_box("Error", msg.get(status, "Error on logout."))
            return
        if status != self.app.current_user:
            util.msg_box("Error", f"You are not the logged-in user ({self.app.current_user}). Logout denied.")
            return
        name = status
        emp_id = name_or_id
        util.msg_box("Goodbye!", f"Goodbye, {name} (ID: {emp_id}).")
        with open(self.log_path, 'a') as f:
            f.write(f'{name},{emp_id},{datetime.datetime.now()},out\n')
        if emp_id in self.app.logged_in_emp_ids:
            self.app.logged_in_emp_ids.remove(emp_id)

        # Reset anti-spoofing tracking on logout
        self.recognition.reset_anti_spoofing()

        self.app.timer_manager.stop()
        self.app.current_user = None
        self.app.reset_ui_after_logout()

    def logout_threaded(self):
        threading.Thread(target=self.logout).start()

