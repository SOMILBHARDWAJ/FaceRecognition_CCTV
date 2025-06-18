import pickle
import threading
import time
import tkinter as tk

import face_recognition
import numpy as np
from PIL import Image, ImageTk
import cv2
import os
import json
import util


class RegistrationHandler:
    def __init__(self, app, recognition_handler):
        self.app = app
        self.recognition = recognition_handler
        self.total_captures = 30
        self.capture_interval = 0.6

    def open_window(self):
        win = tk.Toplevel(self.app.main_window)
        # Position relative to main window
        x = self.app.x_pos + 40
        y = self.app.y_pos + 20
        win.geometry(f"1200x520+{x}+{y}")
        win.title("Register New User")
        # Instruction label
        instruction_text = (
            "📸 Move your face as instructed while we capture:\n"
            "1. Look straight (5 frames)\n"
            "2. Slowly turn left (5 frames)\n"
            "3. Slowly turn right (5 frames)\n"
            "4. Look slightly up/down (5 frames)\n"
            "5. Smile / neutral (10 frames)\n"
            "⚠️ Ensure good lighting and only one face is visible"
        )
        label = tk.Label(win, text=instruction_text, font=("Helvetica", 13), fg="blue", justify="left", wraplength=400, padx=10, pady=10)
        label.place(x=720, y=350)
        # Video feed label
        capture_label = util.get_img_label(win)
        capture_label.place(x=10, y=0, width=700, height=500)
        self.app.webcam.start(capture_label)
        # Entries
        x_label, x_entry = 750, 900
        lbl_id = util.get_text_label(win, 'Emp ID:')
        lbl_id.config(font=("sans-serif", 14))
        lbl_id.place(x=x_label, y=40)
        entry_id = util.get_entry_text(win)
        entry_id.place(x=x_entry, y=40)
        lbl_name = util.get_text_label(win, 'Username:')
        lbl_name.config(font=("sans-serif", 14))
        lbl_name.place(x=x_label, y=100)
        entry_name = util.get_entry_text(win)
        entry_name.place(x=x_entry, y=100)
        btn_accept = util.get_button(win, 'Accept', 'green', lambda: self.accept(win, entry_name, entry_id))
        btn_accept.place(x=850, y=180)
        btn_try = util.get_button(win, 'Try again', 'red', lambda: self.close_window(win))
        btn_try.place(x=850, y=300)
        # Store references
        self.win = win
        self.entry_name = entry_name
        self.entry_id = entry_id
        self.capture_label = capture_label
        self.running = True
        # Start update feed
        self._update_feed()

    def _update_feed(self):
        if not self.running:
            return
        frame = self.app.webcam.get_latest_frame()
        if frame is not None:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            imgtk = ImageTk.PhotoImage(image=pil_img)
            self.capture_label.imgtk = imgtk
            self.capture_label.configure(image=imgtk)
        self.win.after(50, self._update_feed)

    def close_window(self, win):
        self.running = False
        win.destroy()

    def accept(self, win, entry_name, entry_id):
        name = entry_name.get().strip()
        emp_id = entry_id.get().strip()
        if not name or not emp_id:
            util.msg_box("Error", "Name and Emp ID cannot be empty!")
            return
        # Load or init users file
        users_file = os.path.join(self.app.db_dir, 'users.json')
        users_data = {}
        if os.path.exists(users_file) and os.path.getsize(users_file) > 0:
            with open(users_file, 'r') as f:
                try:
                    users_data = json.load(f)
                except json.JSONDecodeError:
                    users_data = {}
        if name in users_data:
            util.msg_box("Error", f"Username '{name}' is already taken!")
            return
        if emp_id in users_data.values():
            util.msg_box("Error", f"Emp ID '{emp_id}' is already registered!")
            return
        # Create user directory
        user_dir = os.path.join(self.app.db_dir, name)
        os.makedirs(user_dir, exist_ok=True)
        # Start capture thread
        threading.Thread(target=self._capture_images, args=(name, emp_id, user_dir)).start()
        self.status_label = tk.Label(win, text="Capturing images...", font=("Helvetica", 12), fg="green")
        self.status_label.place(x=850, y=250)

    def _capture_images(self, name, emp_id, user_dir):
        saved = 0
        encodings = []
        while saved < self.total_captures:
            frame = self.app.webcam.get_latest_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            face_locations = face_recognition.face_locations(frame)
            if len(face_locations) != 1:
                self.win.after(0, lambda: self.status_label.config(text="Ensure only one face is visible"))
                time.sleep(0.5)
                continue
            img_path = os.path.join(user_dir, f'{saved}.jpg')
            cv2.imwrite(img_path, frame)
            face_encs = face_recognition.face_encodings(frame)
            if face_encs:
                encodings.append(face_encs[0])
            saved += 1
            self.win.after(0, lambda count=saved: self.status_label.config(text=f"Capturing image {count}/{self.total_captures}"))
            time.sleep(self.capture_interval)
        # Save user data and encodings
        users_file = os.path.join(self.app.db_dir, 'users.json')
        try:
            with open(users_file, 'r') as f:
                users_data = json.load(f)
        except:
            users_data = {}
        users_data[name] = emp_id
        with open(users_file, 'w') as f:
            json.dump(users_data, f, indent=4)
        # Average encoding
        if encodings:
            avg_encoding = np.mean(encodings, axis=0)
            with open(os.path.join(user_dir, 'avg_encoding.pkl'), 'wb') as f:
                pickle.dump(avg_encoding, f)
        # Save multi encodings
        with open(os.path.join(user_dir, 'multi_encodings.pkl'), 'wb') as f:
            pickle.dump(encodings, f)
        # Reload in recognition handler
        self.recognition.reload_known_faces()
        # Notify
        self.win.after(0, lambda: util.msg_box('Success!', f'User {name} with ID {emp_id} registered successfully!'))
        self.close_window(self.win)
