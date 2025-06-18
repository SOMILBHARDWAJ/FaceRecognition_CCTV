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

        # Define the 5 poses we want to capture
        self.poses = [
            {"name": "Front", "instruction": "Look straight at the camera"},
            {"name": "Left", "instruction": "Turn your head slowly to the LEFT"},
            {"name": "Right", "instruction": "Turn your head slowly to the RIGHT"},
            {"name": "Up", "instruction": "Tilt your head slightly UP"},
            {"name": "Down", "instruction": "Tilt your head slightly DOWN"}
        ]

        self.current_pose_index = 0
        self.capture_interval = 1.5  # Time between captures
        self.captured_encodings = []

    def open_window(self):
        win = tk.Toplevel(self.app.main_window)
        # Position relative to main window
        x = self.app.x_pos + 40
        y = self.app.y_pos + 20
        win.geometry(f"1200x520+{x}+{y}")
        win.title("Register New User - 5 Pose Capture")

        # Instruction label with current pose info
        instruction_text = (
            "📸 POSE-BASED REGISTRATION (5 poses)\n\n"
            "Poses to capture:\n"
            "1. Front - Look straight at camera\n"
            "2. Left - Turn head left\n"
            "3. Right - Turn head right\n"
            "4. Up - Tilt head slightly up\n"
            "5. Down - Tilt head slightly down\n\n"
            "⚠️ Ensure good lighting and only one face visible"
        )

        label = tk.Label(win, text=instruction_text, font=("Helvetica", 11), fg="blue", justify="left", wraplength=400,
                         padx=10, pady=10)
        label.place(x=720, y=300)

        # Status label for current pose
        self.status_label = tk.Label(win, text="Ready to start...", font=("Helvetica", 12, "bold"), fg="green")
        self.status_label.place(x=720, y=250)

        # Progress label
        self.progress_label = tk.Label(win, text="Progress: 0/5", font=("Helvetica", 12), fg="purple")
        self.progress_label.place(x=720, y=270)

        # Video feed label - REUSE existing webcam, don't start new one
        capture_label = util.get_img_label(win)
        capture_label.place(x=10, y=0, width=700, height=500)

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

        btn_accept = util.get_button(win, 'Start Capture', 'green', lambda: self.accept(win, entry_name, entry_id))
        btn_accept.place(x=850, y=180)
        btn_try = util.get_button(win, 'Cancel', 'red', lambda: self.close_window(win))
        btn_try.place(x=850, y=300)

        # Store references
        self.win = win
        self.entry_name = entry_name
        self.entry_id = entry_id
        self.capture_label = capture_label
        self.running = True

        # Start update feed using existing webcam
        self._update_feed()

    def _update_feed(self):
        if not self.running:
            return
        # Get frame from existing webcam manager
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

        # Reset capture state
        self.current_pose_index = 0
        self.captured_encodings = []

        # Start capture thread
        threading.Thread(target=self._capture_poses, args=(name, emp_id, user_dir)).start()

    def _capture_poses(self, name, emp_id, user_dir):
        for pose_idx, pose in enumerate(self.poses):
            # Update status
            self.win.after(0, lambda p=pose, i=pose_idx: self.status_label.config(
                text=f"Pose {i + 1}/5: {p['instruction']}"
            ))
            self.win.after(0, lambda i=pose_idx: self.progress_label.config(
                text=f"Progress: {i}/5"
            ))

            # Wait for user to position
            time.sleep(2)

            # Capture this pose
            attempts = 0
            max_attempts = 10

            while attempts < max_attempts:
                frame = self.app.webcam.get_latest_frame()
                if frame is None:
                    time.sleep(0.1)
                    attempts += 1
                    continue

                face_locations = face_recognition.face_locations(frame)
                if len(face_locations) != 1:
                    self.win.after(0, lambda: self.status_label.config(
                        text="Ensure only one face is visible"
                    ))
                    time.sleep(0.5)
                    attempts += 1
                    continue

                # Good frame - save it
                pose_name = pose['name'].lower()
                img_path = os.path.join(user_dir, f'{pose_name}.jpg')
                cv2.imwrite(img_path, frame)

                # Extract encoding
                face_encs = face_recognition.face_encodings(frame, face_locations)
                if face_encs:
                    self.captured_encodings.append(face_encs[0])
                    # Save individual encoding
                    with open(os.path.join(user_dir, f'{pose_name}_encoding.pkl'), 'wb') as f:
                        pickle.dump(face_encs[0], f)

                self.win.after(0, lambda p=pose: self.status_label.config(
                    text=f"✅ {p['name']} captured!"
                ))
                break

            time.sleep(self.capture_interval)

        # Save all data
        self._save_user_data(name, emp_id, user_dir)

    def _save_user_data(self, name, emp_id, user_dir):
        # Update progress
        self.win.after(0, lambda: self.status_label.config(text="Saving data..."))
        self.win.after(0, lambda: self.progress_label.config(text="Progress: 5/5 - Complete!"))

        # Save user to users.json
        users_file = os.path.join(self.app.db_dir, 'users.json')
        try:
            with open(users_file, 'r') as f:
                users_data = json.load(f)
        except:
            users_data = {}
        users_data[name] = emp_id
        with open(users_file, 'w') as f:
            json.dump(users_data, f, indent=4)

        # Calculate and save average encoding
        if self.captured_encodings:
            avg_encoding = np.mean(self.captured_encodings, axis=0)
            with open(os.path.join(user_dir, 'avg_encoding.pkl'), 'wb') as f:
                pickle.dump(avg_encoding, f)

        # Save multi encodings
        with open(os.path.join(user_dir, 'multi_encodings.pkl'), 'wb') as f:
            pickle.dump(self.captured_encodings, f)

        # Reload in recognition handler
        self.recognition.reload_known_faces()

        # Notify success
        self.win.after(0, lambda: util.msg_box('Success!',
                                               f'User {name} with ID {emp_id} registered successfully with 5 pose captures!'))
        self.win.after(0, lambda: self.close_window(self.win))