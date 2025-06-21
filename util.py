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


def recognize(frame, db_dir, known_encodings=None, known_names=None, use_multi_encodings=False, return_bbox=False):
    """
    Enhanced version with bounding box support for anti-spoofing,
    and fallback to multi-encodings when single-average fails.
    """
    # Convert BGR to RGB for face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Optionally: try a more robust model if needed:
    # face_locations = face_recognition.face_locations(rgb_frame, model='cnn')
    # But default:
    face_locations = face_recognition.face_locations(rgb_frame)

    if len(face_locations) == 0:
        return ('no_persons_found', None, None) if return_bbox else ('no_persons_found', None)
    if len(face_locations) > 1:
        return ('multiple_faces_detected', None, None) if return_bbox else ('multiple_faces_detected', None)

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    if not face_encodings:
        return ('no_persons_found', None, None) if return_bbox else ('no_persons_found', None)

    encoding = face_encodings[0]
    bbox = face_locations[0]  # (top, right, bottom, left)

    if use_multi_encodings:
        # Existing multi-encodings logic (unchanged)
        multi_encodings_dict = {}
        for user in os.listdir(db_dir):
            user_path = os.path.join(db_dir, user)
            if not os.path.isdir(user_path):
                continue
            multi_path = os.path.join(user_path, 'multi_encodings.pkl')
            if os.path.exists(multi_path):
                try:
                    with open(multi_path, 'rb') as f:
                        encs = pickle.load(f)
                        multi_encodings_dict[user] = encs
                except:
                    pass
        matched_user = match_face_multi(encoding, multi_encodings_dict, tolerance=0.5)
        if matched_user != "Unknown":
            # load emp_id as before...
            users_file = os.path.join(db_dir, 'users.json')
            try:
                with open(users_file, 'r') as f:
                    users_data = json.load(f)
                emp_id = users_data.get(matched_user, "N/A")
            except:
                emp_id = "N/A"
            return (matched_user, emp_id, bbox) if return_bbox else (matched_user, emp_id)
        else:
            return ('unknown_person', None, bbox) if return_bbox else ('unknown_person', None)
    else:
        # Original single encoding logic, but with fallback to multi if unknown
        if not known_encodings or not known_names:
            # No known list: skip directly to unknown
            # But we may still try multi-encodings fallback:
            fallback_res = _fallback_multi_if_available(db_dir, encoding, bbox, return_bbox)
            if fallback_res:
                return fallback_res
            return ('unknown_person', None, bbox) if return_bbox else ('unknown_person', None)

        if len(known_encodings) != len(known_names):
            print(f"Warning: Mismatch between encodings ({len(known_encodings)}) and names ({len(known_names)})")
            # try fallback
            fallback_res = _fallback_multi_if_available(db_dir, encoding, bbox, return_bbox)
            if fallback_res:
                return fallback_res
            return ('unknown_person', None, bbox) if return_bbox else ('unknown_person', None)

        try:
            # Possibly adjust tolerance slightly, e.g., 0.45 or so
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.43)
            if any(matches):
                matched_indices = [i for i, match in enumerate(matches) if match]
                if matched_indices:
                    matched_idx = matched_indices[0]
                    if 0 <= matched_idx < len(known_names):
                        matched_user = known_names[matched_idx]
                        users_file = os.path.join(db_dir, 'users.json')
                        try:
                            with open(users_file, 'r') as f:
                                users_data = json.load(f)
                            emp_id = users_data.get(matched_user, "N/A")
                        except:
                            emp_id = "N/A"
                        return (matched_user, emp_id, bbox) if return_bbox else (matched_user, emp_id)
                    else:
                        print(f"Error: Invalid index {matched_idx} for known_names list of length {len(known_names)}")
                        # fall through to fallback
            # If single-encoding didn't match, try fallback to multi-encodings
            fallback_res = _fallback_multi_if_available(db_dir, encoding, bbox, return_bbox)
            if fallback_res:
                return fallback_res
            return ('unknown_person', None, bbox) if return_bbox else ('unknown_person', None)
        except Exception as e:
            print(f"Error in face recognition: {e}")
            # fallback
            fallback_res = _fallback_multi_if_available(db_dir, encoding, bbox, return_bbox)
            if fallback_res:
                return fallback_res
            return ('unknown_person', None, bbox) if return_bbox else ('unknown_person', None)



def load_known_faces(db_path):
    """
    Enhanced to load both average and multi encodings with better error handling
    """
    known_avg_encodings = []
    known_names = []
    multi_encodings_dict = {}

    if not os.path.exists(db_path):
        print(f"Database path {db_path} does not exist")
        return known_avg_encodings, known_names, multi_encodings_dict

    try:
        for user_folder in os.listdir(db_path):
            user_path = os.path.join(db_path, user_folder)
            if not os.path.isdir(user_path):
                continue

            # Load average encoding (for login/logout)
            encoding_path = os.path.join(user_path, 'avg_encoding.pkl')
            if os.path.exists(encoding_path):
                try:
                    with open(encoding_path, 'rb') as f:
                        avg_encoding = pickle.load(f)
                        known_avg_encodings.append(avg_encoding)
                        known_names.append(user_folder)
                        print(f"Loaded average encoding for user: {user_folder}")
                except Exception as e:
                    print(f"Error loading average encoding for {user_folder}: {e}")

            # Load multi-encodings (for timer accuracy)
            multi_path = os.path.join(user_path, 'multi_encodings.pkl')
            if os.path.exists(multi_path):
                try:
                    with open(multi_path, 'rb') as f:
                        multi_encodings = pickle.load(f)
                        multi_encodings_dict[user_folder] = multi_encodings
                        print(f"Loaded multi-encodings for user: {user_folder}")
                except Exception as e:
                    print(f"Error loading multi-encodings for {user_folder}: {e}")

    except Exception as e:
        print(f"Error loading known faces from {db_path}: {e}")

    print(f"Total users loaded: {len(known_names)}")
    print(f"Users with multi-encodings: {len(multi_encodings_dict)}")

    return known_avg_encodings, known_names, multi_encodings_dict

def _fallback_multi_if_available(db_dir, encoding, bbox, return_bbox):
    """
    Try multi-encodings stored for each user if single-average fails.
    Returns tuple if matched, otherwise None.
    """
    # Attempt to load multi_encodings for each user folder
    for user in os.listdir(db_dir):
        user_path = os.path.join(db_dir, user)
        if not os.path.isdir(user_path):
            continue
        multi_path = os.path.join(user_path, 'multi_encodings.pkl')
        if os.path.exists(multi_path):
            try:
                with open(multi_path, 'rb') as f:
                    enc_list = pickle.load(f)
            except:
                continue
            # Try matching against each encoding in list
            # You may choose a tolerance here, e.g., 0.5 or slightly higher
            matches = face_recognition.compare_faces(enc_list, encoding, tolerance=0.53)
            if any(matches):
                # Found a match in multi-encodings
                # Load emp_id
                users_file = os.path.join(db_dir, 'users.json')
                try:
                    with open(users_file, 'r') as f:
                        users_data = json.load(f)
                    emp_id = users_data.get(user, "N/A")
                except:
                    emp_id = "N/A"
                return (user, emp_id, bbox) if return_bbox else (user, emp_id)
    # No fallback match
    return None
