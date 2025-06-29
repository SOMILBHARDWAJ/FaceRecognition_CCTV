import cv2
from PIL import Image, ImageTk


class WebcamManager:
    def __init__(self, camera_index=0, update_interval=20):
        self.camera_index = camera_index
        self.update_interval = update_interval
        self.cap = None
        self.frame = None
        self.running = False
        self.label = None

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
            # Convert and display
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            imgtk = ImageTk.PhotoImage(image=pil_img)
            # Avoid garbage collection
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
        # Schedule next update
        self.label.after(self.update_interval, self._update_frame)

    def get_latest_frame(self):
        return self.frame