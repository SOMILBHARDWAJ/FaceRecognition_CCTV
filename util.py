import os
import json
import tkinter as tk
from tkinter import messagebox
import face_recognition
import cv2
import numpy as np
import pickle

def match_face(current_encoding, known_encodings, known_names, tolerance=0.43):
    if not known_encodings:
        return "Unknown"

    face_distances = face_recognition.face_distance(known_encodings, current_encoding)
    best_match_index = np.argmin(face_distances)

    if face_distances[best_match_index] < tolerance:
        return known_names[best_match_index]
    else:
        return "Unknown"


def match_face_multi(current_encoding, multi_encodings_dict, tolerance=0.53):
    for name, encodings in multi_encodings_dict.items():
        if not encodings:
            continue
        matches = face_recognition.compare_faces(encodings, current_encoding, tolerance)
        if any(matches):
            return name
    return "Unknown"

def get_button(window, text, color, command, fg='white'):
    return tk.Button(
        window, text=text, fg=fg, bg=color,
        activebackground="black", activeforeground="white",
        command=command, height=2, width=20,
        font=('Helvetica bold', 20)
    )


def get_img_label(window):
    label = tk.Label(window)
    label.grid(row=0, column=0)
    return label


def get_text_label(window, text):
    return tk.Label(window, text=text, font=("sans-serif", 21), justify="left")


def get_entry_text(window):
    return tk.Entry(window, font=("Arial", 20))


def msg_box(title, description):
    messagebox.showinfo(title, description)

def recognize(frame, db_dir, known_encodings=None, known_names=None, use_multi_encodings=False):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    if len(face_locations) == 0:
        return 'no_persons_found', None
    if len(face_locations) > 1:
        return 'multiple_faces_detected', None

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    if not face_encodings:
        return 'no_persons_found', None

    encoding = face_encodings[0]

    if use_multi_encodings:
        all_encodings = []
        all_names = []
        for user in os.listdir(db_dir):
            user_path = os.path.join(db_dir, user)
            if not os.path.isdir(user_path):
                continue
            for file in os.listdir(user_path):
                if file.endswith('.pkl') and file != 'avg_encoding.pkl':
                    file_path = os.path.join(user_path, file)
                    try:
                        with open(file_path, 'rb') as f:
                            enc = pickle.load(f)
                            all_encodings.append(enc)
                            all_names.append(user)
                    except:
                        continue

        if not all_encodings:
            return 'unknown_person', None

        matches = face_recognition.compare_faces(all_encodings, encoding, tolerance=0.53)
        if np.any(matches):
            matched_idx = np.argmax(matches)
            matched_user = all_names[matched_idx]
            users_file = os.path.join(db_dir, 'users.json')
            with open(users_file, 'r') as f:
                users_data = json.load(f)
            return matched_user, users_data.get(matched_user, "N/A")
        else:
            return 'unknown_person', None

    else:
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.43)
        if np.any(matches):
            matched_idx = matches.index(True)
            matched_user = known_names[matched_idx]
            users_file = os.path.join(db_dir, 'users.json')
            with open(users_file, 'r') as f:
                users_data = json.load(f)
            return matched_user, users_data.get(matched_user, "N/A")
        else:
            return 'unknown_person', None

def load_known_faces(db_path):
    known_avg_encodings = []
    known_names = []
    multi_encodings_dict = {}

    for user_folder in os.listdir(db_path):
        user_path = os.path.join(db_path, user_folder)
        if not os.path.isdir(user_path):
            continue

        encoding_path = os.path.join(user_path, 'avg_encoding.pkl')
        if os.path.exists(encoding_path):
            with open(encoding_path, 'rb') as f:
                avg_encoding = pickle.load(f)
                known_avg_encodings.append(avg_encoding)
                known_names.append(user_folder)

        # Load multi-encodings
        multi_path = os.path.join(user_path, 'multi_encodings.pkl')
        if os.path.exists(multi_path):
            with open(multi_path, 'rb') as f:
                multi_encodings = pickle.load(f)
                multi_encodings_dict[user_folder] = multi_encodings

    return known_avg_encodings, known_names, multi_encodings_dict

def match_face_multi(current_encoding, multi_encodings_dict, tolerance=0.42):
    """
    Matches against multiple encodings stored per user.
    Returns matched name or 'Unknown'.
    """
    for name, encodings in multi_encodings_dict.items():
        if not encodings:
            continue
        matches = face_recognition.compare_faces(encodings, current_encoding, tolerance)
        if any(matches):
            return name
    return "Unknown"
