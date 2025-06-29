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

        # UI elements for better control
        self.pose_indicator = None
        self.progress_label = None
        self.pose_indicators = []

        self.registration_started = False
        self.current_name = None
        self.current_emp_id = None
        self.current_user_dir = None
        self.btn_capture = None
        self.btn_accept = None  # Store reference to start button

    def check_face_already_registered(self, test_frame, tolerance=0.32):
        """
        Check if the face in the test frame is already registered in the system
        Returns: (is_duplicate, existing_user_name, existing_emp_id) or (False, None, None)
        """
        try:
            # Extract face encoding from test frame
            rgb_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if len(face_locations) == 0:
                return False, None, None, "No face detected"
            elif len(face_locations) > 1:
                return False, None, None, "Multiple faces detected"

            test_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if not test_encodings:
                return False, None, None, "Could not extract face encoding"

            test_encoding = test_encodings[0]

            # Load users data to get name-empid mapping
            users_file = os.path.join(self.app.db_dir, 'users.json')
            users_data = {}
            if os.path.exists(users_file):
                with open(users_file, 'r') as f:
                    try:
                        users_data = json.load(f)
                    except json.JSONDecodeError:
                        users_data = {}

            # Check against all registered users
            for user_folder in os.listdir(self.app.db_dir):
                user_path = os.path.join(self.app.db_dir, user_folder)
                if not os.path.isdir(user_path) or user_folder.endswith('.json'):
                    continue

                # Try to load average encoding first
                avg_encoding_path = os.path.join(user_path, 'avg_encoding.pkl')
                if os.path.exists(avg_encoding_path):
                    try:
                        with open(avg_encoding_path, 'rb') as f:
                            existing_encoding = pickle.load(f)

                        # Compare faces
                        face_distance = face_recognition.face_distance([existing_encoding], test_encoding)[0]

                        if face_distance < tolerance:
                            emp_id = users_data.get(user_folder, "Unknown")
                            return True, user_folder, emp_id, None

                    except Exception as e:
                        print(f"Error loading encoding for {user_folder}: {e}")

                # Also check against multi-encodings for more accuracy
                multi_encoding_path = os.path.join(user_path, 'multi_encodings.pkl')
                if os.path.exists(multi_encoding_path):
                    try:
                        with open(multi_encoding_path, 'rb') as f:
                            multi_encodings = pickle.load(f)

                        # Check against all multi-encodings
                        matches = face_recognition.compare_faces(multi_encodings, test_encoding, tolerance=tolerance)
                        if any(matches):
                            emp_id = users_data.get(user_folder, "Unknown")
                            return True, user_folder, emp_id, None

                    except Exception as e:
                        print(f"Error loading multi-encodings for {user_folder}: {e}")

            return False, None, None, None

        except Exception as e:
            return False, None, None, f"Error during face comparison: {str(e)}"

    def open_window(self):
        win = tk.Toplevel(self.app.main_window)
        x = self.app.x_pos + 40
        y = self.app.y_pos + 20
        win.geometry(f"1400x600+{x}+{y}")  # Increased size
        win.title("Register New User - 5 Pose Capture")
        win.resizable(False, False)
        win.configure(bg='#f0f0f0')  # Light background

        # Main container frames
        # Left frame for webcam
        left_frame = tk.Frame(win, bg='#f0f0f0', relief='ridge', bd=2)
        left_frame.place(x=10, y=10, width=750, height=520)

        # Right frame for controls
        right_frame = tk.Frame(win, bg='#ffffff', relief='ridge', bd=2)
        right_frame.place(x=780, y=10, width=600, height=520)

        # Webcam feed with border
        webcam_title = tk.Label(left_frame, text="📹 Live Camera Feed",
                                font=("Helvetica", 14, "bold"), bg='#f0f0f0', fg='#333333')
        webcam_title.place(x=10, y=10)

        capture_label = util.get_img_label(left_frame)
        capture_label.place(x=20, y=40, width=700, height=460)
        capture_label.configure(bg='black', relief='sunken', bd=2)

        # Status section in right frame
        status_title = tk.Label(right_frame, text="📊 Registration Status",
                                font=("Helvetica", 16, "bold"), bg='#ffffff', fg='#2c3e50')
        status_title.place(x=20, y=15)

        # Current pose indicator with larger, more visible design
        self.pose_indicator = tk.Label(right_frame, text="🎯 Ready to Start",
                                       font=("Helvetica", 14, "bold"),
                                       bg='#3498db', fg='white',
                                       relief='raised', bd=3, padx=10, pady=5)
        self.pose_indicator.place(x=20, y=50, width=560, height=40)

        # Progress bar simulation
        self.progress_frame = tk.Frame(right_frame, bg='#ffffff')
        self.progress_frame.place(x=20, y=100, width=560, height=30)

        self.progress_label = tk.Label(self.progress_frame, text="Progress: 0/5 poses completed",
                                       font=("Helvetica", 11), bg='#ffffff', fg='#7f8c8d')
        self.progress_label.pack(side='left')

        # Visual progress indicators
        self.pose_indicators = []
        indicator_frame = tk.Frame(right_frame, bg='#ffffff')
        indicator_frame.place(x=20, y=140, width=560, height=60)

        pose_names = ["Front", "Left", "Right", "Up", "Down"]
        for i, pose_name in enumerate(pose_names):
            indicator = tk.Label(indicator_frame, text=f"{i + 1}. {pose_name}",
                                 font=("Helvetica", 10), bg='#ecf0f1', fg='#7f8c8d',
                                 relief='ridge', bd=1, padx=8, pady=4)
            indicator.place(x=i * 110, y=10, width=100, height=40)
            self.pose_indicators.append(indicator)

        # User information section
        info_title = tk.Label(right_frame, text="👤 User Information",
                              font=("Helvetica", 14, "bold"), bg='#ffffff', fg='#2c3e50')
        info_title.place(x=20, y=220)

        # Employee ID
        lbl_id = tk.Label(right_frame, text='Employee ID:', font=("Helvetica", 12),
                          bg='#ffffff', fg='#34495e')
        lbl_id.place(x=20, y=260)
        entry_id = tk.Entry(right_frame, font=("Arial", 12), relief='ridge', bd=2, width=25)
        entry_id.place(x=150, y=260)

        # Username
        lbl_name = tk.Label(right_frame, text='Username:', font=("Helvetica", 12),
                            bg='#ffffff', fg='#34495e')
        lbl_name.place(x=20, y=300)
        entry_name = tk.Entry(right_frame, font=("Arial", 12), relief='ridge', bd=2, width=25)
        entry_name.place(x=150, y=300)

        # Instructions section
        instructions_title = tk.Label(right_frame, text="📋 Instructions",
                                      font=("Helvetica", 14, "bold"), bg='#ffffff', fg='#2c3e50')
        instructions_title.place(x=20, y=350)

        instruction_text = (
            "1. Enter your Employee ID and Username\n"
            "2. Click 'Start Registration' to begin\n"
            "3. Follow the pose instructions shown above\n"
            "4. Click 'Capture Photo' for each pose\n"
            "5. Keep your face clearly visible\n\n"
            "⚠️  Ensure good lighting and only one face visible\n"
            "⚠️  Face verification will check for duplicates"
        )

        instructions_label = tk.Label(right_frame, text=instruction_text,
                                      font=("Helvetica", 10), fg="#2c3e50", bg='#ffffff',
                                      justify="left", anchor="nw")
        instructions_label.place(x=20, y=380, width=540, height=100)

        # Buttons with better styling
        self.btn_accept = tk.Button(right_frame, text='🚀 Start Registration',
                                    font=("Helvetica", 12, "bold"), bg='#27ae60', fg='white',
                                    relief='raised', bd=3, padx=20, pady=8,
                                    activebackground='#229954', activeforeground='white',
                                    command=lambda: self.accept(win, entry_name, entry_id))
        self.btn_accept.place(x=50, y=490)

        # NEW: Capture Photo Button (initially hidden)
        self.btn_capture = tk.Button(right_frame, text='📸 Capture Photo',
                                     font=("Helvetica", 12, "bold"), bg='#f39c12', fg='white',
                                     relief='raised', bd=3, padx=20, pady=8,
                                     activebackground='#e67e22', activeforeground='white',
                                     command=self.capture_current_pose)
        # Initially place it off-screen (will be moved when registration starts)
        self.btn_capture.place(x=-200, y=490)

        btn_cancel = tk.Button(right_frame, text='❌ Cancel',
                               font=("Helvetica", 12, "bold"), bg='#e74c3c', fg='white',
                               relief='raised', bd=3, padx=20, pady=8,
                               activebackground='#c0392b', activeforeground='white',
                               command=lambda: self.close_window(win))
        btn_cancel.place(x=450, y=490)

        # Store references
        self.win = win
        self.entry_name = entry_name
        self.entry_id = entry_id
        self.capture_label = capture_label
        self.running = True
        self.registration_started = False

        # Start webcam feed
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

        # NEW: Check if face is already registered
        self.pose_indicator.config(
            text="🔍 Checking for face duplicates...",
            bg='#f39c12', fg='white'
        )
        self.win.update()  # Force UI update

        current_frame = self.app.webcam.get_latest_frame()
        if current_frame is None:
            util.msg_box("Error", "Unable to capture frame for verification. Please try again.")
            self.pose_indicator.config(
                text="🎯 Ready to Start",
                bg='#3498db', fg='white'
            )
            return

        is_duplicate, existing_name, existing_emp_id, error_msg = self.check_face_already_registered(current_frame)

        if error_msg:
            util.msg_box("Error", f"Face verification failed: {error_msg}")
            self.pose_indicator.config(
                text="🎯 Ready to Start",
                bg='#3498db', fg='white'
            )
            return

        if is_duplicate:
            duplicate_message = (
                f"❌ Face Already Registered!\n\n"
                f"This person is already registered in the system:\n\n"
                f"Existing Username: {existing_name}\n"
                f"Existing Employee ID: {existing_emp_id}\n\n"
                f"Each person can only register once in the system.\n"
                f"If you need to update your information, please contact your administrator."
            )
            util.msg_box("Registration Denied", duplicate_message)
            self.pose_indicator.config(
                text="❌ Face already registered",
                bg='#e74c3c', fg='white'
            )
            return

        # If we reach here, face is not duplicate - proceed with registration
        self.pose_indicator.config(
            text="✅ Face verification passed",
            bg='#27ae60', fg='white'
        )
        self.win.update()
        time.sleep(1)  # Brief pause to show success

        # Create user directory
        user_dir = os.path.join(self.app.db_dir, name)
        os.makedirs(user_dir, exist_ok=True)

        # Reset capture state
        self.current_pose_index = 0
        self.captured_encodings = []
        self.registration_started = True
        self.current_name = name
        self.current_emp_id = emp_id
        self.current_user_dir = user_dir

        # Hide start button and show capture button using stored reference
        self.btn_accept.place(x=-200, y=490)  # Hide start button
        self.btn_capture.place(x=250, y=490)  # Show capture button

        # Disable entry fields
        entry_name.config(state='disabled')
        entry_id.config(state='disabled')

        # Show first pose instruction
        self.update_pose_indicator(0, "active")

    def capture_current_pose(self):
        """NEW METHOD: Capture photo for current pose when button is clicked"""
        if not self.registration_started:
            return

        if self.current_pose_index >= len(self.poses):
            util.msg_box("Error", "All poses have been captured!")
            return

        # Get current frame
        frame = self.app.webcam.get_latest_frame()
        if frame is None:
            util.msg_box("Error", "Unable to capture frame. Please try again.")
            return

        # Check for face
        face_locations = face_recognition.face_locations(frame)

        if len(face_locations) == 0:
            util.msg_box("Error", "No face detected. Please position yourself in front of camera and try again.")
            return
        elif len(face_locations) > 1:
            util.msg_box("Error", "Multiple faces detected. Ensure only one person is visible and try again.")
            return

        # Show capturing feedback
        current_pose = self.poses[self.current_pose_index]
        self.pose_indicator.config(
            text=f"📸 Capturing {current_pose['name']} pose...",
            bg='#3498db', fg='white'
        )
        self.win.update()  # Force UI update

        # Capture and save
        try:
            # Save the image
            pose_name = current_pose['name'].lower()
            img_path = os.path.join(self.current_user_dir, f'{pose_name}.jpg')
            cv2.imwrite(img_path, frame)

            # Extract encoding
            face_encs = face_recognition.face_encodings(frame, face_locations)
            if face_encs:
                self.captured_encodings.append(face_encs[0])
                # Save individual encoding
                with open(os.path.join(self.current_user_dir, f'{pose_name}_encoding.pkl'), 'wb') as f:
                    pickle.dump(face_encs[0], f)

            # Show success feedback
            self.update_pose_indicator(self.current_pose_index, "captured")

            # Move to next pose
            self.current_pose_index += 1

            if self.current_pose_index < len(self.poses):
                # Show next pose after brief delay
                self.win.after(1000, lambda: self.update_pose_indicator(self.current_pose_index, "active"))
            else:
                # All poses captured, save data
                self.btn_capture.config(state='disabled', text='✅ All Photos Captured')
                self.win.after(1000, self._save_user_data_manual)

        except Exception as e:
            util.msg_box("Error", f"Failed to capture photo: {str(e)}")
            # Reset the pose indicator
            self.update_pose_indicator(self.current_pose_index, "active")

    def _save_user_data_manual(self):
        """NEW METHOD: Save user data after manual capture"""
        # Show saving progress
        self.update_pose_indicator(0, "saving")

        # Save user to users.json
        users_file = os.path.join(self.app.db_dir, 'users.json')
        try:
            with open(users_file, 'r') as f:
                users_data = json.load(f)
        except:
            users_data = {}

        users_data[self.current_name] = self.current_emp_id
        with open(users_file, 'w') as f:
            json.dump(users_data, f, indent=4)

        # Calculate and save average encoding
        if self.captured_encodings:
            avg_encoding = np.mean(self.captured_encodings, axis=0)
            with open(os.path.join(self.current_user_dir, 'avg_encoding.pkl'), 'wb') as f:
                pickle.dump(avg_encoding, f)

        # Save multi encodings
        with open(os.path.join(self.current_user_dir, 'multi_encodings.pkl'), 'wb') as f:
            pickle.dump(self.captured_encodings, f)

        # Reload in recognition handler
        self.recognition.reload_known_faces()

        # Show completion
        self.update_pose_indicator(0, "complete")

        # Show success message with more details
        success_message = (
            f"🎉 Registration Successful!\n\n"
            f"User: {self.current_name}\n"
            f"Employee ID: {self.current_emp_id}\n"
            f"Poses Captured: 5/5\n\n"
            f"You can now use the system for attendance tracking."
        )

        util.msg_box('Registration Complete!', success_message)
        self.close_window(self.win)

    def update_pose_indicator(self, pose_index, status):
        """Update the pose indicator and progress display"""
        if status == "active":
            # Show current pose instruction
            if pose_index < len(self.poses):
                current_pose = self.poses[pose_index]
                self.pose_indicator.config(
                    text=f"🎯 {current_pose['instruction']}",
                    bg='#f39c12', fg='white'
                )

                # Update progress
                self.progress_label.config(text=f"Progress: {pose_index}/5 poses completed")

                # Update pose indicator visuals
                for i, indicator in enumerate(self.pose_indicators):
                    if i == pose_index:
                        indicator.config(bg='#3498db', fg='white')  # Current pose
                    elif i < pose_index:
                        indicator.config(bg='#27ae60', fg='white')  # Completed
                    else:
                        indicator.config(bg='#ecf0f1', fg='#7f8c8d')  # Not started

        elif status == "captured":
            # Show success for captured pose
            current_pose = self.poses[pose_index]
            self.pose_indicator.config(
                text=f"✅ {current_pose['name']} pose captured successfully!",
                bg='#27ae60', fg='white'
            )

            # Update progress
            self.progress_label.config(text=f"Progress: {pose_index + 1}/5 poses completed")

            # Update indicator to green
            if pose_index < len(self.pose_indicators):
                self.pose_indicators[pose_index].config(bg='#27ae60', fg='white')

        elif status == "saving":
            self.pose_indicator.config(
                text="💾 Saving registration data...",
                bg='#9b59b6', fg='white'
            )

        elif status == "complete":
            self.pose_indicator.config(
                text="🎉 Registration completed successfully!",
                bg='#27ae60', fg='white'
            )
            self.progress_label.config(text="Progress: 5/5 poses completed - DONE!")

            # Mark all indicators as complete
            for indicator in self.pose_indicators:
                indicator.config(bg='#27ae60', fg='white')